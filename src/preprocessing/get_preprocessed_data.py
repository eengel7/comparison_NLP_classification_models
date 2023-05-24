from src.preprocessing.preprocessor import LogRegPreprocessor, TransformerPreprocessor
from typing import Tuple, Any


def get_preprocessed_data(model_type: str, overwrite_data: bool = True, random_seed: int = None, select_level_of_classification: bool = False, args = None, language : str = 'en', split_val: bool = True) -> Tuple[Any]:
    if model_type == 'logistic_regression':
        preprocessor = LogRegPreprocessor(model_type, overwrite_data, random_seed, select_level_of_classification, args, language, split_val)
    else:
        preprocessor = TransformerPreprocessor(model_type, overwrite_data, random_seed, select_level_of_classification, args, language, split_val)
        
    return preprocessor.get_preprocessed_data() 


