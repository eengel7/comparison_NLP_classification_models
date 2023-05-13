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

# Experiment config:----------------------
model_type = 'bert'
model_name = "bert-base-uncased" 

for random_seed in [40,41]:
# for random_seed in [40,41,42,43,44]:

    # Prepare data
    data_args = DataArgs(random_seed = random_seed)
    print(data_args)
    X_train, X_test, X_val, Y_train, Y_test, Y_val = get_preprocessed_data(model_type, overwrite_data= False, args = data_args)
    train_df = prepare_df(X_train, Y_train)
    val_df = prepare_df(X_val, Y_val)
    test_df = prepare_df(X_test, Y_test)

    # Initialise the model
    model_args = MultiLabelClassificationArgs(
                                            manual_seed = random_seed,
                                            wandb_kwargs = {"name":f'{model_name}_{random_seed}'},

                                            wandb_project ='multi-label-BERT', 
                                            learning_rate = 5e-5,
                                            num_train_epochs=15,
                                            train_batch_size = 4,
                                            eval_batch_size = 4,
                                            evaluate_during_training= True, 
                                            use_multiprocessing= True,
                                            use_early_stopping= True,
                                            early_stopping_patience=3,
                                            early_stopping_delta= 0
                                            ) 
    
    model = MultiLabelClassificationModel(
        model_type,
        model_name,
        num_labels=305,
        args=model_args,
        use_cuda = True
    )

    # Training
    print('Data is loaded. Training starts.')
    train_model(model, train_df, eval_df = val_df)

    # Evaluation
    result, model_outputs, wrong_predictions = eval_model(model,
        test_df
    )
    print(result)
