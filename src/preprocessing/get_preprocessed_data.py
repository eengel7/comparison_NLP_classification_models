from src.preprocessing.preprocessor import LogRegPreprocessor, BERTPreprocessor
import pandas as pd
from config.data_args import DataArgs 


def get_preprocessed_data(model_type: str, overwrite_data: bool = True, select_level_of_classification: bool = False, args = None):
    if model_type == 'logistic_regression':
        preprocessor = LogRegPreprocessor(model_type, overwrite_data, select_level_of_classification, args)
        
    elif model_type in ['bert','roberta']:
         preprocessor = BERTPreprocessor(model_type, overwrite_data, select_level_of_classification, args)
    
    else:
        print("Model type is not found.")
        return 
    
    preprocessor.get_preprocessed_data() 
    

