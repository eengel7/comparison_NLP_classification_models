import os
import pickle
from abc import ABC, abstractmethod

from omegaconf import DictConfig

from typing import DataFrame
import hydra
import numpy as np
import pandas as pd

class Preprocessor(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @abstractmethod
    def preprocess_and_store(self):
        pass

    def load_dataframe(self):
        df = pd.read_csv(hydra.utils.to_absolute_path(self.cfg.run_mode.paths.raw_data))
        return df

    def get_x_y_texts(self, df: DataFrame):
        x = df[self.cfg.pre_processing.token_type]
        texts = df[self.cfg.text_col]
        return x, y, texts

    def store_data(self, data, dest_dir, file_name):
        dest_dir = hydra.utils.to_absolute_path(dest_dir)
        os.makedirs(dest_dir, exist_ok=True)
        pickle.dump(data, open(os.path.join(dest_dir, file_name), "wb"))
        print(f"Saved {file_name} at {dest_dir}.")

class LogRegPreprocessor(Preprocessor):
    def __init__(self):
        path_to_tfidf = hydra.utils.to_absolute_path(self.cfg.run_mode.paths.tfidf_weights)

        self.tfidf_weights = np.load(
            os.path.join(path_to_tfidf, "word2weight_idf.npy"), allow_pickle=True
        ).item()
        assert isinstance(self.tfidf_weights, dict)

        self.max_idf = np.load(
            os.path.join(path_to_tfidf, "max_idf.npy"),
            allow_pickle=True,
        )

    def preprocess_and_store(self):
        df = self.load_dataframe()
        if self.cfg.run_mode.augment:
            df = replace_with_gendered_pronouns(self.cfg.run_mode.augment, self.cfg.text_col, df)
        df = self.basic_tokenize(df)
        model = get_embedding(self.cfg)
        vectorizer = self.get_vectorizer(model)

        dev_set, test_set = get_dev_test_sets(self.cfg, self.cfg.label_col, df)
        for split_df, split_name in zip([dev_set, test_set], ["dev_split", "test_split"]):
            x, y, texts = self.get_x_y_texts(split_df)
            x = vectorizer.transform(x)
            self.store_data(
                {"X": x, "Y": y, "texts": texts},
                get_data_dir(self.cfg),
                split_name,
            )

    def basic_tokenize(self, df):
        sgt = SimpleTokenizer(
            ("german" if self.cfg.language == "GER" else "english"),
            self.cfg.run_mode.tokenize.to_lower,
            self.cfg.run_mode.tokenize.remove_punctuation,
        )
        # tokenize
        df = sgt.tokenize(df, text_col=self.cfg.text_col)
        return df

    def get_vectorizer(self, model):
        if self.cfg.pre_processing.mean:
            vectorizer = MeanEmbeddingVectorizer(
                model, self.tfidf_weights, max_idf=self.max_idf
            )
        else:
            vectorizer = WordEmbeddingVectorizer(
                model,
                self.tfidf_weights,
                max_idf=self.max_idf,
                seq_length=self.cfg.pre_processing.seq_length,
            )
        return         