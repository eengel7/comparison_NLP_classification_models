from abc import ABC, abstractmethod

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class Vectorizer(ABC):
    # def __init__(self):
    #     self.cfg = cfg

    @abstractmethod
    def transform(self):
        pass


class BowVectorizer(Vectorizer):
    def __init__(self, args):
        self.args = args
        self.count_vect = CountVectorizer(ngram_range=(1,2), max_features = self.args.max_feaures) 
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        word_embs = self.count_vect.fit_transform(df) 
        return word_embs

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        word_embs = self.count_vect.transform(df)
        return word_embs