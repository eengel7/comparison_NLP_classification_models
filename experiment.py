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
                                          wandb_project ='multi_label_transformer', 
                                          wandb_kwargs = {"name": model_name},
                                          learning_rate = 5e-5,
                                          num_train_epochs=15,
                                          train_batch_size = 4,
                                          eval_batch_size = 4,
                                          ) 
model = MultiLabelClassificationModel(
    model_type,
    model_name,
    num_labels=305,
    args=model_args,
)
# -----------------------------------
# Prepare data
X_train, X_test, X_val, Y_train, Y_test, Y_val = get_preprocessed_data(model_type, overwrite_data= True)

train_df = prepare_df(X_train, Y_train)
test_df = prepare_df(X_test, Y_test)
val_df = prepare_df(X_val, Y_val)
print('Data is loaded.')

train_model(model, train_df, eval_df = val_df)

# # Evaluate the model
# result, model_outputs, wrong_predictions = eval_model(model,
#     test
# )

#print(result, model_outputs, wrong_predictions)
# # Make predictions with the model
# predictions, raw_outputs = model.predict(["Sam"])

# print(predictions,raw_outputs)
