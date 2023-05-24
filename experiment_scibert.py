import logging

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
import wandb
wandb.login()

if __name__ == "__main__":


    # Experiment config:----------------------
    model_type = 'auto'
    model_name = "allenai/scibert_scivocab_uncased" 

    for random_seed in [43,44]:
        print('----------------------------------------Random seed:',{random_seed})

    # for random_seed in [40,41,42,43,44]:
        # Prepare data
        data_args = DataArgs(random_seed = random_seed)
        X_train, X_test, X_val, Y_train, Y_test, Y_val = get_preprocessed_data(model_type, overwrite_data= False)
        train_df = prepare_df(X_train, Y_train)
        val_df = prepare_df(X_val, Y_val)
        test_df = prepare_df(X_test, Y_test)


        # Initialise the model
        name_project = f'{model_name}_{random_seed}'
        model_args = MultiLabelClassificationArgs(
                                                wandb_project ='multi-label-DistilBERT', 
                                                manual_seed = random_seed,
                                                wandb_kwargs = {"name": name_project},
                                                learning_rate = 5e-5,
                                                num_train_epochs=15,
                                                train_batch_size = 4,
                                                eval_batch_size = 4,
                                                evaluate_during_training= True, 
                                                use_multiprocessing= False,
                                                use_early_stopping= True,
                                                early_stopping_patience=3,
                                                early_stopping_delta= 0
                                                ) 

        model = MultiLabelClassificationModel( 
            model_type,
            model_name,
            num_labels=305,
            args=model_args,
            use_cuda = False
        )

        # Training
        print('Training starts.')
        train_model(model, train_df, eval_df = val_df)

        # Evaluation
        result, model_outputs, wrong_predictions = eval_model(model,
            test_df
        )
        print(result)
