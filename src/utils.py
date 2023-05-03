import torch
import warnings
from torch.nn import CrossEntropyLoss
import numpy as np
import pandas as pd



def calculate_loss(model, inputs, num_labels, weight, device):

    if weight:
        loss_fct = CrossEntropyLoss(weight=torch.Tensor(weight).to(device))
    else:
        loss_fct = None
        
    outputs = model(**inputs)
    # model outputs are always tuple in pytorch-transformers (see doc)
    loss = outputs[0]
    if loss_fct:
        logits = outputs[1]
        labels = inputs["labels"]

        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    return (loss, *outputs[1:])


def prepare_df(features: pd.Series, targets: np.ndarray):
    targets = pd.Series(targets.tolist())
    df = pd.DataFrame(features)
    df['labels'] = targets.values
    return df