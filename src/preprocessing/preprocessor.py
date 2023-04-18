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
from sklearn.preprocessing import MultiLabelBinarizer, normalize

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


        if self.select_level_of_classification:
            info = f'{self.args.digits}_level'
        else:
            info = 'all_levels'

        self.file_name = f"preprocessed_data_{self.language}_{info}.pkl"


    @abstractmethod
    def get_preprocessed_data(self):
        pass

    def data_exists(self) -> bool:
        dest = self.args.preprocessed_path.join(self.model_type)
        os.makedirs(dest, exist_ok=True)
        file = os.path.join(dest, self.file_name) 
        if  os.path.isfile(file):
            print(f"{self.file_name} already exists at {dest}.")
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
            research_fields = research_fields.split('|')  
            
            # Get unique research fields and normalise strings
            research_fields = list({" ".join(x.lower().strip().split()) for x in research_fields if x})
            
            class_IDs = list()
            # Match research field with given SCB classification and return classification with specific level, i.e. number of digits
            for field in research_fields:
                
                class_id = df_classifications.index[df_classifications[f'Label_{self.language}'] == field][:] 
                if field =='unclassified':
                    pass
                elif not class_id.empty:
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

        use_columns = [f'title_{self.language}', f'description_{self.language}', f'research_fields_{self.language}']
        df = pd.read_excel(self.args.raw_data_path, usecols = use_columns)
        df.rename(columns = {f'research_fields_{self.language}':'labels'}, inplace = True)

        if join_headline_body_by_line_break:
            df["text"] = df[f'title_{self.language}'].astype(str) +"\n"+ df[f'description_{self.language}']
        else:
            df["text"] = df[f'title_{self.language}'].astype(str) + df[f'description_{self.language}']

        df = self.clean_df(df)  
        return df[['text','labels']]

    def clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Remove nan
        df = df.dropna().reset_index(drop=True)

        # Remove unclassified data
        df = df[df['labels'] != 'Unclassified']

        word_counts = df[['text']].apply(lambda x: len(' '.join(x).split()), axis=1)
        df = df[word_counts >= GlobalArgs.min_words]

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
    
    def prepare_targets(self, Y_train, Y_test):
        le = MultiLabelBinarizer()
        Y_train = le.fit_transform(Y_train)
        Y_test = le.transform(Y_test)

        print(f'{len(le.classes_)} classes were encoded by MultiLabelBinarizer.')

        return Y_train, Y_test

    def store_data(self, dataset_dict: dict):
        dest = self.args.preprocessed_path + self.model_type
        os.makedirs(dest, exist_ok=True)
        file = os.path.join(dest, self.file_name) 

        if not os.path.isfile(file):
            pickle.dump(dataset_dict, open(file, "wb"))
            print(f"Saved {self.file_name} at {dest}.")

        elif os.path.isfile(file) and self.overwrite_data:
            pickle.dump(dataset_dict, open(file, "wb"))
            print(f"{self.file_name} at {dest} is overwritten.")
            
        else:
            print(f"{self.file_name} already exists at {dest}.")
        
    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = DataArgs()
        args.load(input_dir)
        return args
    
class LogRegPreprocessor(Preprocessor):

    def get_preprocessed_data(self):

        if self.data_exists and not self.overwrite_data:
            print('Data already exists and will not be overwritten.')
            return 
        
        # Check whether data already exists
        elif self.data_exists and self.overwrite_data:
            print('Data already exists and will be overwritten.')

        # Prepare data
        df = self.load_dataframe()
        
        simple_tokenizer = SimpleTokenizer(remove_stopwords =  self.args.remove_stopwords, apply_stemming =  self.args.apply_stemming) 
        df['text'] = simple_tokenizer(df['text'])

        # Load SCB classifications and retrieve unique label for data
        df_classifications = self.load_classifications()
        df['labels'] = df['labels'].apply(lambda x: self.get_matching_SCB_class_IDs(x, df_classifications))

        # Filter labels depending on chosen level of classification (in config)
        if self.select_level_of_classification:
            df['labels'] = df['labels'].apply(lambda x: self.filter_IDs(x)) 

        # Split data
        X_train, X_test, Y_train, Y_test = self.split_data(df)
        Y_train, Y_test = self.prepare_targets(Y_train, Y_test)
 

        # Vectorize tokens based on BOW embedding 
        vectorizer = BowVectorizer()
        X_train = vectorizer.fit_transform(X_train) 
        X_test = vectorizer.transform(X_test)
        print(X_train)
        # count_vect = CountVectorizer(ngram_range=(1,2))

        # X_train = count_vect.fit_transform(X_train) 
        # X_test= count_vect.transform(X_test)

        # X_train = normalize(X_train)
        # X_test = normalize(X_test)

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

        # Split data and encode labels
        X_train, X_test, Y_train, Y_test = self.split_data(df)
        Y_train, Y_test = self.prepare_targets(Y_train, Y_test)
        Y_train = pd.Series(Y_train.tolist()) 
        Y_test = pd.Series(Y_test.tolist()) 

        dataset_dict = {"X_train": X_train, "X_test": X_test, "Y_train": Y_train, "Y_test": Y_test}
        self.store_data(dataset_dict)

  