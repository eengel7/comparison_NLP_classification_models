import os
import pickle
from abc import ABC, abstractmethod
import random
import numpy as np
import joblib
from config.data_args import DataArgs

import pandas as pd
from typing import Any, Tuple
from src.preprocessing.tokenizer import SimpleTokenizer
from src.preprocessing.vectorizer import BowVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, normalize

class Preprocessor(ABC):
    def __init__(
        self,
        model_type: str,
        overwrite_data: bool = True,
        random_seed: int = None,
        select_level_of_classification: bool = False,
        args = None,
        language: str = None,
        split_val: bool = True,
    ):
        """
        Initializes a Preprocessor.

        Args: 
            model_type: The type of model (bert, xlnet, xlm, roberta, distilbert)
            model_name: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            tokenizer_type: The type of tokenizer (auto, bert, xlnet, xlm, roberta, distilbert, etc.) to use. If a string is passed, Simple Transformers will try to initialize a tokenizer class from the available MODEL_CLASSES.
                                Alternatively, a Tokenizer class (subclassed from PreTrainedTokenizer) can be passed.
        """
        self.args = DataArgs()

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, DataArgs):
            self.args = args

        self.model_type = model_type
        self.overwrite_data = overwrite_data
        self.select_level_of_classification = select_level_of_classification
        
        if isinstance(random_seed, int):
            self.random_seed = random_seed
        elif isinstance(self.args.random_seed, int):
            self.random_seed = self.args.random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        

        if language:
            self.language = language
        else:
            self.language = self.args.language

        

        self.split_val = split_val
        if self.split_val:
            split_info = '_val'
        
        if self.select_level_of_classification:
            info = f'{self.args.digits}_level'
        else:
            info = 'all_levels'

        self.file_name = 'preprocessed_data.pkl'
        if model_type == 'logistic_regression':
            dir_name = f"logistic_regression_{self.language}_{info}{split_info}_{self.random_seed}"
        else:
            dir_name = f"transformer_{self.language}_{info}{split_info}_{self.random_seed}"
        self.dest_name = self.args.preprocessed_path + dir_name
        self.binarizer_name = f"binarizer_{info}.joblib"

    @abstractmethod
    def get_preprocessed_data(self) -> Tuple[Any]:
        pass

    def data_exists(self) -> bool:
        os.makedirs(self.dest_name, exist_ok=True)
        file = os.path.join(self.dest_name, self.file_name) 
        if  os.path.isfile(file):
            print(f"{self.file_name} already exists at {self.dest_name}.")
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
                if field in ['unclassified', 'oklassificerad']:
                    pass
                elif not class_id.empty:
                    class_IDs.extend(class_id.values[:])
                else:
                    print(f"No matching classification found for {field}.")
    
            return class_IDs
        else:
            print(f"{research_fields} is not a string.")
            return None
        
    def guarantee_hierarchical_path(self, IDs: list[int])-> list[int]:

        IDs = IDs.copy()
        for label in IDs:
            digits = len(str(label))

            if digits >= 3:
                first_digit = int(str(label)[0])
                if not first_digit in IDs:
                    IDs.append(first_digit)
            elif digits == 5:
                three_digits = int(str(label)[0:3])
                if not three_digits in IDs:
                    IDs.append(three_digits)
        return IDs
                    
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
        df = df[word_counts >= self.args.min_words]

        return df

    def split_data(self, df: pd.DataFrame, get_val: bool = True) -> list[pd.DataFrame]:
        test_size = self.args.test_size
        validation_size = self.args.validation_size 
        print(f'Random seed {self.random_seed} is used for splitting the data.')
        X_train, X_test, Y_train, Y_test  = train_test_split(
            df['text'],
            df['labels'],
            test_size = test_size,
            shuffle = True,
            random_state= self.random_seed
        )

        if get_val:
            X_train, X_val, Y_train, Y_val  = train_test_split(
                X_train,
                Y_train,
                test_size = validation_size,
                shuffle = True,
                random_state= self.random_seed
            )

            return X_train, X_test, X_val, Y_train, Y_test, Y_val
        return X_train, X_test, Y_train, Y_test
    
    def prepare_targets(self, Y_train, Y_test, Y_val = None):
        os.makedirs(self.dest_name, exist_ok=True)
        file = os.path.join(self.dest_name, self.binarizer_name) 
        
        if os.path.isfile(file):
            # load binarizer from file
            le = joblib.load(file)
        else:
            le = MultiLabelBinarizer()
            le.fit(Y_train)
            # save binarizer to file
            print(f'Saving new binarizer to {file}...')
            joblib.dump(le, file)

        Y_train = le.transform(Y_train)
        Y_test = le.transform(Y_test)
        if isinstance(Y_val, pd.Series):
            Y_val = le.transform(Y_val) 

        print(f'{len(le.classes_)} classes were encoded by MultiLabelBinarizer.')

        return Y_train, Y_test, Y_val

    def store_data(self, data: Any):
        os.makedirs(self.dest_name, exist_ok=True)
        file = os.path.join(self.dest_name, self.file_name) 

        if not os.path.isfile(file):
            pickle.dump(data, open(file, "wb"))
            self.save_data_args(self.dest_name)
            print(f"Saved {self.file_name} at {self.dest_name}.")

        elif os.path.isfile(file) and self.overwrite_data:
            pickle.dump(data, open(file, "wb"))
            self.save_data_args(self.dest_name)
            print(f"{self.file_name} at {self.dest_name} is overwritten.")
            
        else:
            print(f"{self.file_name} already exists at {self.dest_name}.")
        
    def save_data_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = DataArgs()
        args.load(input_dir)
        return args
    
class LogRegPreprocessor(Preprocessor):

    def get_preprocessed_data(self):

        if self.data_exists() and not self.overwrite_data:
            print('Data already exists and will not be overwritten.')
            os.makedirs(self.dest_name, exist_ok=True)
            file = os.path.join(self.dest_name, self.file_name) 
            with open(file, 'rb') as file:
                dataset_dict = pickle.load(file)
            
            if self.split_val:
                X_train, X_test, X_val, Y_train, Y_test, Y_val = dataset_dict["X_train"], dataset_dict["X_test"], dataset_dict["X_val"] ,dataset_dict["Y_train"], dataset_dict["Y_test"], dataset_dict["Y_val"]
                return X_train, X_test, X_val, Y_train, Y_test, Y_val
            
            else:
                X_train, X_test, Y_train, Y_test = dataset_dict["X_train"], dataset_dict["X_test"], dataset_dict["Y_train"], dataset_dict["Y_test"]
                return X_train, X_test, Y_train, Y_test
        
        # Check whether data already exists
        elif self.data_exists() and self.overwrite_data:
            print('Data already exists but will be overwritten.')

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
        

        # Add parenting label nodes to match hiearchical structure
        if self.args.add_parent_nodes:
            nof_labels = df['labels'].str.len().sum()
            df['labels'] = df['labels'].apply(lambda x: self.guarantee_hierarchical_path(x))
            #print(f'{df["labels"].str.len().sum()- nof_labels} labels are added to the already existing {nof_labels} to match the hierarchical structure.')

        if self.split_val:
            # Split data and encode labels
            X_train, X_test, X_val, Y_train, Y_test, Y_val = self.split_data(df, get_val = True)
            Y_train, Y_test, Y_val = self.prepare_targets(Y_train, Y_test, Y_val)

            # Vectorize tokens based on BOW embedding 
            vectorizer = BowVectorizer(self.args)
            X_train = vectorizer.fit_transform(X_train) 
            X_val = vectorizer.transform(X_val)
            X_test = vectorizer.transform(X_test)

            dataset_dict = {"X_train": X_train, "X_test": X_test, "X_val": X_val,"Y_train": Y_train, "Y_test": Y_test, "Y_val": Y_val}
            self.store_data(dataset_dict)

            return X_train, X_test, X_val, Y_train, Y_test, Y_val
        
        else:
            # Split data and encode labels
            X_train, X_test,  Y_train, Y_test = self.split_data(df, get_val = False)
            Y_train, Y_test, _= self.prepare_targets(Y_train, Y_test)
             # Vectorize tokens based on BOW embedding 
            vectorizer = BowVectorizer(self.args)
            X_train = vectorizer.fit_transform(X_train) 
            X_test = vectorizer.transform(X_test)

            dataset_dict = {"X_train": X_train, "X_test": X_test, "Y_train": Y_train, "Y_test": Y_test}
            self.store_data(dataset_dict)

            return X_train, X_test, Y_train, Y_test
        


class TransformerPreprocessor(Preprocessor):
    
    def get_preprocessed_data(self):

        # Check whether data already exists
        if self.data_exists() and not self.overwrite_data:
            print('Data already exists and will not be overwritten.')
            os.makedirs(self.dest_name, exist_ok=True)
            file = os.path.join(self.dest_name, self.file_name) 
            with open(file, 'rb') as file:
                dataset_dict = pickle.load(file)
            
            if self.split_val:
                X_train, X_test, X_val, Y_train, Y_test, Y_val = dataset_dict["X_train"], dataset_dict["X_test"], dataset_dict["X_val"] ,dataset_dict["Y_train"], dataset_dict["Y_test"], dataset_dict["Y_val"]
                return X_train, X_test, X_val, Y_train, Y_test, Y_val
            
            else:
                X_train, X_test, Y_train, Y_test = dataset_dict["X_train"], dataset_dict["X_test"], dataset_dict["Y_train"], dataset_dict["Y_test"]
                return X_train, X_test, Y_train, Y_test

        
         # Check whether data already exists
        elif self.data_exists() and self.overwrite_data:
            print('Data already exists but will be overwritten.')

        df = self.load_dataframe()

        # Load SCB classifications and retrieve unique label for data
        df_classifications = self.load_classifications()
        df['labels'] = df['labels'].apply(lambda x: self.get_matching_SCB_class_IDs(x, df_classifications))
        # Filter labels depending on chosen level of classification (in config)
        if self.select_level_of_classification:
            df['labels'] = df['labels'].apply(lambda x: self.filter_IDs(x)) 

        

        # Add parenting label nodes to match hiearchical structure
        if self.args.add_parent_nodes:
            nof_labels = df['labels'].str.len().sum()
            df['labels'] = df['labels'].apply(lambda x: self.guarantee_hierarchical_path(x))
            # print(f'{df["labels"].str.len().sum()- nof_labels} labels are added to the already existing {nof_labels} to match the hierarchical structure.')
       

        if self.split_val:
            # Split data and encode labels
            X_train, X_test, X_val, Y_train, Y_test, Y_val = self.split_data(df, get_val = True)
            Y_train, Y_test, Y_val = self.prepare_targets(Y_train, Y_test, Y_val)
            dataset_dict = {"X_train": X_train, "X_test": X_test, "X_val": X_val,"Y_train": Y_train, "Y_test": Y_test, "Y_val": Y_val}
            self.store_data(dataset_dict)

            return X_train, X_test, X_val, Y_train, Y_test, Y_val
        
        else:
            # Split data and encode labels
            X_train, X_test,  Y_train, Y_test = self.split_data(df, get_val = False)
            Y_train, Y_test, _= self.prepare_targets(Y_train, Y_test)
            dataset_dict = {"X_train": X_train, "X_test": X_test, "Y_train": Y_train, "Y_test": Y_test}
            self.store_data(dataset_dict)

            return X_train, X_test, Y_train, Y_test

  