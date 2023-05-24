from src.preprocessing.get_preprocessed_data import get_preprocessed_data
import logging

import wandb
from config.data_args import DataArgs
from src.classification import (MultiLabelClassificationArgs,
                                MultiLabelClassificationModel)
from src.evaluation import eval_model
from src.preprocessing.get_preprocessed_data import get_preprocessed_data
from src.training import train_model
from src.utils import prepare_df
import pandas as pd


X_train, X_test, X_val, Y_train, Y_test, Y_val = get_preprocessed_data('bert', overwrite_data= False, random_seed= 42)  


# Initialise the model
model_args = MultiLabelClassificationArgs(
                                        learning_rate = 5e-5,
                                        num_train_epochs=15,
                                        train_batch_size = 4,
                                        use_multiprocessing= False,
                                        use_early_stopping= True,
                                        early_stopping_patience=3,
                                        early_stopping_delta= 0,
                                        max_seq_length= 128
                                        ) 

model = MultiLabelClassificationModel( 
    'bert',
    'bert-base-uncased',
    num_labels=305,
    args= model_args, 
    use_cuda = False,
    
)
train_df = prepare_df(X_train, Y_train)
val_df = prepare_df(X_val, Y_val)
test_df = prepare_df(X_test, Y_test) 

combined_df = pd.concat([train_df, val_df, test_df], axis=0)
# Training
print('Training starts.')
train_model(model, combined_df)