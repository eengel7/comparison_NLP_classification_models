from src.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs
)
import pandas as pd
import logging
from src.training import train_model
from src.evaluation import eval_model
from src.prediction import predict
from sklearn import metrics
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_data = [
    ["Aragorn", [1, 0, 0]],
    ["Frodo", [0, 1, 1]],
    ["Gimli", [1, 0, 1]],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]

# Preparing eval data
eval_data = [
    ["Legolas", [1, 0, 0]],
    ["Merry", [0, 0, 1]],
    ["Eomer", [1, 0, 0]],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]

# Optional model configuration
model_args = MultiLabelClassificationArgs(num_train_epochs=2, use_multiprocessing = False,
                                          wandb_project ='test_run', 
                                          wandb_kwargs = {"name": 'bert'},
                                          evaluate_during_training= True)


# Create a MultiLabelClassificationModel
model = MultiLabelClassificationModel(
    "roberta",
    "roberta-base",
    num_labels=3,
    args=model_args,
    
)

# Train the model ,f1_score_micro = metrics.f1_score(average='micro'), f1_score_macro = metrics.f1_score(average='macro')
train_model(model, train_df, eval_df = eval_df)

# Evaluate the model
result, model_outputs, wrong_predictions = eval_model(model,
    eval_df,
)

# # Make predictions with the model

predictions, raw_outputs = model.predict(["Sam"])

print(predictions,raw_outputs)
