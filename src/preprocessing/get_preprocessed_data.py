from src.preprocessing.preprocessor import LogRegPreprocessor, BERTPreprocessor
from typing import Tuple, Any


def get_preprocessed_data(model_type: str, overwrite_data: bool = True, select_level_of_classification: bool = False, args = None, language : str = 'en', split_val: bool = True) -> Tuple[Any]:
    if model_type == 'logistic_regression':
        preprocessor = LogRegPreprocessor(model_type, overwrite_data, select_level_of_classification, args, language, split_val)
        
    elif model_type in ['bert','roberta']:
        preprocessor = BERTPreprocessor(model_type, overwrite_data, select_level_of_classification, args, language, split_val)
    
    else:
        print("Model type is not found.")
        return None
        
    return preprocessor.get_preprocessed_data() 


