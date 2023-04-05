import os
import pickle
from abc import ABC, abstractmethod
import random
import numpy as np

from config.global_args import GlobalArgs
from config.data_args import DataArgs

import pandas as pd
from typing import Any
from src.preprocessing.tokenizer import SimpleTokenizer
from src.preprocessing.vectorizer import BowVectorizer
from sklearn.model_selection import train_test_split

class Preprocessor(ABC):
    def __init__(
        self,
        model_type: str,
        overwrite_data: bool = True,
        select_level_of_classification: bool = False,
        args = None,
        language: str = None,
    ):
        """
        Initializes a Preprocessor.

        Args: 
            model_type: The type of model (bert, xlnet, xlm, roberta, distilbert)
            model_name: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            tokenizer_type: The type of tokenizer (auto, bert, xlnet, xlm, roberta, distilbert, etc.) to use. If a string is passed, Simple Transformers will try to initialize a tokenizer class from the available MODEL_CLASSES.
                                Alternatively, a Tokenizer class (subclassed from PreTrainedTokenizer) can be passed.
        """
        # self.args = self._load_model_args(model_type), TODO: check or just delete this part
        self.args = DataArgs()
        self.model_type = model_type
        self.overwrite_data = overwrite_data
        self.select_level_of_classification = select_level_of_classification
        
        if GlobalArgs.manual_seed:
            self.random_seed = GlobalArgs.random_seed
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        if language:
            self.language = language
        else:
            self.language = self.args.language

        if isinstance(args, dict):
            self.args.update_from_dict(args)


    @abstractmethod
    def get_preprocessed_data(self):
        pass

    def data_exists(self) -> bool:
        dest = self.args.preprocessed_path.join(self.model_type)
        file = os.path.join(dest, f"preprocessed_data.pkl") 
        if  os.path.isfile(file):
            print(f"preprocessed_data.pkl already exists at {dest}.")
            return True
        else:
            return False
            

    def load_classifications(self) -> pd.DataFrame:
        df_classifications = pd.read_excel(self.args.raw_classifications_SCB,  index_col=0)

        # Normalise labels
        df_classifications[f'Label_{self.language}'] = df_classifications[f'Label_{self.language}'].apply(lambda x: x.strip().lower())

        return df_classifications
    
    

    def get_matching_SCB_class_IDs(self, research_fields: str, df_classifications: pd.DataFrame) -> list[int]:
        # retrieve class ID for all research fields'
        
        if isinstance(research_fields, str):
            research_fields = research_fields.split('; ')  
            
            # Get unique research fields and normalise strings
            research_fields = list({" ".join(x.lower().strip().split()) for x in research_fields if x})
            
            class_IDs = list()
            # Match research field with given SCB classification and return classification with specific level, i.e. number of digits
            for field in research_fields:
                
                class_id = df_classifications.index[df_classifications[f'Label_{self.language}'] == field][:] 
                if not class_id.empty:
                    class_IDs.extend(class_id.values[:])
                else:
                    print(f"No matching classification found for {field}.")
    
            return class_IDs
        else:
            print(f"{research_fields} is not a string.")
            return None

    def filter_IDs(self, IDs: list[int]) -> list[int]:
        # select IDs depending on the provided classification level 
        return [id for id in IDs if len(str(id)) == self.args.digits]


    def load_dataframe(self, join_headline_body_by_line_break: bool = True) -> pd.DataFrame:

        use_columns = [f'Title {self.language}', f'Description {self.language}', 'Research fields']
        
        df = pd.read_excel(self.args.raw_data_path, usecols = use_columns)
        df = self.clean_df(df)

        df.rename(columns = {'Research fields':'labels'}, inplace = True)
        

        if join_headline_body_by_line_break:
            df["text"] = df[f'Title {self.language}'].astype(str) +"\n"+ df[f'Description {self.language}']
        else:
            df["text"] = df[f'Title {self.language}'].astype(str) + df[f'Description {self.language}']
            
        return df[['text','labels']]

    def clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Remove nan
        df = df.dropna().reset_index(drop=True)

        # Remove noise from data TODO: add other noise and depending on language
        df['Description En'] = df['Description En'].apply(lambda x: x.replace('Purpose and goal: ','' ))

        # Remove unclassified data
        df = df[df['Research fields'] != 'Unclassified']

        return df

    def split_data(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        test_size = self.args.test_size

        X_train, X_test, Y_train, Y_test  = train_test_split(
            df['text'],
            df['labels'],
            test_size = test_size,
            shuffle = True,
            #stratify = df['Research fields'],
        )

        return X_train, X_test, Y_train, Y_test

    def store_data(self, dataset_dict: dict):
        dest = self.args.preprocessed_path + self.model_type
        os.makedirs(dest, exist_ok=True)
        file = os.path.join(dest, f"preprocessed_data.pkl") 
        if not os.path.isfile(file):
            pickle.dump(dataset_dict, open(file, "wb"))
            print(f"Saved preprocessed_data.pkl at {dest}.")
        else:
            print(f"preprocessed_data.pkl already exists at {dest}.")
        
    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = DataArgs()
        args.load(input_dir)
        return args
    
class LogRegPreprocessor(Preprocessor):

    def get_preprocessed_data(self):
        # TODO: redo this section and include choice of language 

        if self.data_exists and not self.overwrite_data:
            print('Data already exists and will not be overwritten.')
            return 
        # Check whether data already exists
        elif self.data_exists and self.overwrite_data:
            print('Data already exists and will be overwritten.')

        # Prepare data
        df = self.load_dataframe()
        
        simple_tokenizer = SimpleTokenizer(remove_stopwords =  self.cfg.run_mode.tokenize.remove_stopwords, apply_stemming =  self.cfg.run_mode.tokenize.apply_stemming) 
        df[f'Description {self.language}'] = simple_tokenizer(df[f'Description {self.language}'])

        # Load SCB classifications and retrieve unique label for data
        df_classifications = self.load_classifications()
        df['Research fields'] = df['Research fields'].apply(lambda x: self.get_matching_SCB_class_IDs(x, df_classifications))

        # Filter labels depending on  chosen level of classification (in config)
        df['Research fields'] = df['Research fields'].apply(lambda x: self.filter_IDs(x)) 

        print(df)

        # Split data
        X_train, X_test, Y_train, Y_test = self.split_data(df)

        # Vectorize tokens based on BOW embedding 
        vectorizer = BowVectorizer()
        X_train = vectorizer.fit_transform(X_train) 
        X_test = vectorizer.transform(X_test)

        dataset_dict = {"X_train": X_train, "X_test": X_test, "Y_train": Y_train, "Y_test": Y_test}

        self.store_data(dataset_dict)

class BERTPreprocessor(Preprocessor):
    
    def get_preprocessed_data(self):

        # Check whether data already exists
        if self.data_exists and not self.overwrite_data:
            print('Data already exists and will not be overwritten.')
            return 
        
        elif self.data_exists and self.overwrite_data:
            print('Data already exists and will be overwritten.')

        df = self.load_dataframe()

        # Load SCB classifications and retrieve unique label for data
        df_classifications = self.load_classifications()
        df['labels'] = df['labels'].apply(lambda x: self.get_matching_SCB_class_IDs(x, df_classifications))

        # Filter labels depending on chosen level of classification (in config)
        if self.select_level_of_classification:
            df['labels'] = df['labels'].apply(lambda x: self.filter_IDs(x)) 

        # Split data
        X_train, X_test, Y_train, Y_test = self.split_data(df)

        dataset_dict = {"X_train": X_train, "X_test": X_test, "Y_train": Y_train, "Y_test": Y_test}
        self.store_data(dataset_dict)

  