import os

import hydra
import hydra.utils
import numpy as np

from joblib import dump, load
from sklearn.linear_model import LogisticRegression

from src.classifier.classifiers import RegardLSTM, RegardBERT


def create_classifier(cfg):
    if cfg.model.name == "logistic_regression":
        classifier = LogisticRegression(
        )
    else:
        print( "Models implemented so far: Logistic regression, ")

    return classifier