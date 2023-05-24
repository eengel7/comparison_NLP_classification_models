import logging

import wandb
from config.data_args import DataArgs
from src.classification import (MultiLabelClassificationArgs,
                                MultiLabelClassificationModel)
from src.evaluation import eval_model
from src.preprocessing.get_preprocessed_data import get_preprocessed_data
from src.training import train_model
from src.utils import prepare_df

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
wandb.login()

if __name__ == "__main__":

    # Experiment config:----------------------
    model_type = 'bert'
    model_name = "bert-base-uncased" 

    for random_seed in [45,46]:
        print('----------------------------------------Random seed:',{random_seed})

        name_run = f'{model_name}_{random_seed}'
        wandb_project = f'multi-label-{model_name}_128'
        wandb_kwargs = {"name": name_run}

        # Prepare data
        data_args = DataArgs(random_seed = random_seed)
        X_train, X_test, X_val, Y_train, Y_test, Y_val = get_preprocessed_data(model_type, overwrite_data= False, random_seed= random_seed)
        train_df = prepare_df(X_train, Y_train)
        val_df = prepare_df(X_val, Y_val)
        test_df = prepare_df(X_test, Y_test)

        # Initialise the model
        model_args = MultiLabelClassificationArgs(
                                                wandb_project = wandb_project, 
                                                manual_seed = random_seed,
                                                wandb_kwargs = wandb_kwargs,
                                                learning_rate = 5e-5,
                                                num_train_epochs=15,
                                                train_batch_size = 4,
                                                eval_batch_size = 4,
                                                evaluate_during_training= True, 
                                                use_multiprocessing= False,
                                                use_early_stopping= True,
                                                early_stopping_patience=3,
                                                early_stopping_delta= 0,
                                                max_seq_length=128
                                                ) 

        model = MultiLabelClassificationModel( 
            model_type,
            model_name,
            num_labels=305,
            args=model_args,
            use_cuda = True
        )

        # Training
        print('Training starts.')
        train_model(model, train_df, eval_df = val_df)

        # Evaluation
        result, model_outputs, wrong_predictions = eval_model(model,
            test_df
        )

        # Add data and prediction to wandb 
        run = wandb.init(
                        project=wandb_project,
                        job_type="split-dataset",
                        name = name_run,
                        resume=True

        )
        # log the data as an artifact
        data_artifact = wandb.Artifact("data", "dataset")
        data_artifact.add_file(f"data/preprocessed/transformer_en_all_levels_val_{random_seed}/preprocessed_data.pkl")
        run.log_artifact(data_artifact)
        

        # log the data config as an artifact
        config_artifact = wandb.Artifact("config", type="config")
        config_artifact.add_file(f"data/preprocessed/transformer_en_all_levels_val_{random_seed}/data_args.json")
        run.log_artifact(config_artifact)

        # Log the predictions
        run.log({"test_predictions": str(model_outputs.tolist())})

        # Log the metrics
        run.log({"test_LRAP": result["LRAP"], "test_f1_score_avg": result["f1_score_avg"], "test_f1_score_macro": result["f1_score_macro"], "test_f1_score_micro": result["f1_score_micro"]})

        # finish logging the data logging run
        run.finish()
