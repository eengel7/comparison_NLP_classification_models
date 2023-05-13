from src.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs
)
import pandas as pd
import logging
import pickle
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from src.training import train_model
from src.evaluation import eval_model
from src.preprocessing.get_preprocessed_data import get_preprocessed_data
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier 

from src.utils import prepare_df
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Experiment config:----------------------
model_type = 'bert'
model_name = "bert-base-uncased" # 'bert-base-uncased' ,"roberta-base", "outputs/checkpoint-120-epoch-1",

model_args = MultiLabelClassificationArgs(
                                          wandb_project ='multi-label-42_predictions', 
                                          wandb_kwargs = {"name": model_name},
                                          learning_rate = 5e-5,
                                          num_train_epochs=10,
                                          train_batch_size = 4,
                                          eval_batch_size = 4,
                                          evaluate_during_training= True, 
                                          use_multiprocessing= True,
                                          use_early_stopping= True,
                                          early_stopping_patience=3,
                                          early_stopping_delta= 1e-5
                                          ) 
model = MultiLabelClassificationModel(
    model_type,
    'best_model',
    num_labels=305,
    args=model_args
)
# -----------------------------------
# Prepare data
X_train, X_test, X_val, Y_train, Y_test, Y_val = get_preprocessed_data('bert', overwrite_data= True)
# test_df = prepare_df(X_test, Y_test)
# # Evaluate the model
# result, model_outputs, wrong_predictions = eval_model(model, eval_df=test_df.head(10)
# )
