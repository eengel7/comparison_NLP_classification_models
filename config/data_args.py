import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from multiprocessing import cpu_count
import warnings

from torch.utils.data import Dataset


@dataclass
class DataArgs():
    """
    Model args for Preprocessor
    """

    remove_structure_headlines: bool = True
    # Loading raw data
    raw_data_path: str = 'data/raw_data/ymner_data.xlsx' # Funded projects
    raw_classifications_SCB: str = 'data/raw_data/SCB_classifications.xlsx' # SCB based classifications
    
    preprocessed_path: str = 'data/preprocessed/'
    language: str ='en'   # 'En' or 'Sv'
    test_size: int= 0.2
    validation_size: int = 0.2   # fraction used from training 
    add_parent_nodes: bool = True

    # Logistic regression
    remove_stopwords: bool = True
    apply_stemming: bool = True
    
    # SCB classifications 
    digits: int =  5   # Options: 1, 3,5        Number of digits indicating the level of classification

    def update_from_dict(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))
    
    def get_args_for_saving(self):
        args_for_saving = {
            key: value
            for key, value in asdict(self).items()
        }
        return args_for_saving

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "data_args.json"), "w") as f:
            args_dict = self.get_args_for_saving()
            json.dump(args_dict, f)

    def load(self, input_dir):
        if input_dir:
            data_args_file = os.path.join(input_dir, "data_args.json")
            if os.path.isfile(data_args_file):
                with open(data_args_file, "r") as f:
                    model_args = json.load(f)

                self.update_from_dict(model_args)