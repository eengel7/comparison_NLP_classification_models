import os
import pickle
from abc import ABC, abstractmethod

import hydra
import pandas as pd
from omegaconf import DictConfig
from typing import Any
from src.preprocessing.tokenizer import SimpleTokenizer
from src.preprocessing.vectorizer import BowVectorizer
from sklearn.model_selection import train_test_split

class Preprocessor(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @abstractmethod
    def preprocess_and_store(self):
        pass

    def load_classifications(self) -> pd.DataFrame:
        df_classifications = pd.read_excel(hydra.utils.to_absolute_path(self.cfg.run_mode.paths.classifications),  index_col=0)

        # Normalise labels
        df_classifications['Label_En'] = df_classifications['Label_En'].apply(lambda x: x.strip().lower())

        return df_classifications
    
    def get_matching_SCB_classification_ID(self, research_fields: str, df_classifications: pd.DataFrame) -> str:
        # retrieve unique ID for classification of data'

        if isinstance(research_fields, str):
            research_fields = research_fields.split(', ')   #TODO: need to take care of single classifications that contain comma

            # Get unique research fields and normalise strings
            research_fields = list({x.lower().strip() for x in research_fields if x})
            
            # Match research field with given SCB classification and return classification with specific level, i.e. number of digits
            for field in research_fields:
                classification_id = df_classifications.index[df_classifications['Label_En'] == field][0]
            
                level_of_classification = len(str(classification_id))
                if level_of_classification == self.cfg.run_mode.digits:
                    return classification_id
                else:
                    print(f"No matching classification found for {research_fields}.")
                    return None
                
        else:
            return None


    def load_dataframe(self) -> pd.DataFrame:
        use_columns = ['Title En', 'Description En', 'Research fields']
        df = pd.read_excel(hydra.utils.to_absolute_path(self.cfg.run_mode.paths.raw_data), usecols = use_columns)
        print('Raw data is loaded.')
        return df

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:

        # Remove nan
        df = df.dropna().reset_index(drop=True)

        # Remove noise from data
        df['Description En'] = df['Description En'].apply(lambda x: x.replace('Purpose and goal: ','' ))

        # Remove unclassified data
        df = df[df['Research fields'] != 'Unclassified']

        return df

    def split_data(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        test_size = self.cfg.run_mode.test_size

        X_train, X_test, Y_train, Y_test  = train_test_split(
            df['Description En'],
            df['Research fields'],
            test_size = test_size,
            shuffle = True,
            stratify = df['Research fields'],
            random_state = 42,
        )

        return X_train, X_test, Y_train, Y_test

    def store_data(self, data: Any, file_name: str):
        dest = self.cfg.run_mode.paths.preprocessed_path
        os.makedirs(dest, exist_ok=True)

        dest = hydra.utils.to_absolute_path(dest)
        #TODO add to check whether data exists or not 

        file = os.path.join(dest, f"{file_name}.pkl") 
        if not os.path.isfile(file):
            pickle.dump(data, open(file, "wb"))
            print(f"Saved {file_name} at {dest}.")
        else:
            print(f"{file_name} already exists at {dest}.")

class LogRegPreprocessor(Preprocessor):

    def preprocess_and_store(self):

        # Prepare data
        df = self.load_dataframe()
        df = self.filter_dataframe(df)
        simple_tokenizer = SimpleTokenizer(remove_stopwords =  self.cfg.run_mode.tokenize.remove_stopwords, apply_stemming =  self.cfg.run_mode.tokenize.apply_stemming) 
        df['Description En'] = simple_tokenizer(df['Description En'])

        # Load SCB classifications and retrieve unique label for data
        df_classifications = self.load_classifications()
        df['Research fields'] = df['Research fields'].apply(lambda x: self.get_matching_SCB_classification_ID(x, df_classifications)) 
        print(df)

        # Split data
        X_train, X_test, Y_train, Y_test = self.split_data(df)

        # Vectorize tokens based on BOW embedding 
        vectorizer = BowVectorizer()
        X_train_bow = vectorizer.fit_transform(X_train) 
        X_test_bow = vectorizer.transform(X_test)

        for data, file_name in zip([X_train_bow, X_test_bow, Y_train, Y_test ], ["X_train", "X_test", "Y_train", "Y_test"]):
            self.store_data(data, file_name)