from src.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs
)
import pandas as pd
import logging
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from src.training import train_model
from src.evaluation import eval_model
from src.preprocessing.get_preprocessed_data import get_preprocessed_data
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier 


from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from sklearn import preprocessing
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Experiment config:
model_type = 'bert'
model_name = "bert-base-uncased" # 'bert-base-uncased' ,"roberta-base", "outputs/checkpoint-120-epoch-1",

model_args = MultiLabelClassificationArgs(num_train_epochs=1, 
                                          use_multiprocessing = False, 
                                          wandb_project ='test_sample1', 
                                          wandb_kwargs = {"name": model_type})
model = MultiLabelClassificationModel(
    model_type,
    model_name,
    args=model_args,
    num_labels=234,

)

# Prepare data
# Output first five rows
# get_preprocessed_data(model_type)



with open('/Users/evaengel/comparison_NLP_classification_models/data/preprocessed_for_bert/preprocessed_data.pkl', 'rb') as file:
    dataset_dict = pickle.load(file)
X_train, X_test, Y_train, Y_test = dataset_dict["X_train"], dataset_dict["X_test"], dataset_dict["Y_train"], dataset_dict["Y_test"]




train = pd.DataFrame(X_train)
train['labels'] = Y_train.values

test = pd.DataFrame(X_test)
test['labels'] = Y_test.values
print(train)
print(test)


# Train the model
train_model(model, train, eval_df = test)

# # Evaluate the model
# result, model_outputs, wrong_predictions = eval_model(model,
#     test
# )

#print(result, model_outputs, wrong_predictions)
# # Make predictions with the model
# predictions, raw_outputs = model.predict(["Sam"])

# print(predictions,raw_outputs)
