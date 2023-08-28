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
    language: str ='en'   # 'en' or 'sv'
    test_size: int= 0.2
    validation_size: int = 0.2   # fraction used from training 
    add_parent_nodes: bool = True

    # Logistic regression
    remove_stopwords: bool = True
    apply_stemming: bool = True
    min_words: int = 20
    random_seed: int = 42
    max_feaures: int = 10000
    
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

    
    level_1_digit = [1, 2, 3, 4, 5, 6] 
    level_3_digits = [101, 102, 103, 104, 105, 106, 107, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 301, 302, 303, 304, 305, 401, 402, 403, 404, 405, 501, 502, 503, 504, 505, 506, 507, 508, 509, 601, 602, 603, 604, 605] 
    level_5_digits = [10101, 10102, 10103, 10104, 10105, 10106, 10199, 10201, 10202, 10203, 10204, 10205, 10206, 10207, 10208, 10209, 10299, 10301, 10302, 10303, 10304, 10305, 10306, 10399, 10401, 10402, 10403, 10404, 10405, 10406, 10407, 10499, 10501, 10502, 10503, 10504, 10505, 10506, 10507, 10508, 10509, 10599, 10601, 10602, 10603, 10604, 10605, 10606, 10607, 10608, 10609, 10610, 10611, 10612, 10613, 10614, 10615, 10699, 10799, 20101, 20102, 20103, 20104, 20105, 20106, 20107, 20108, 20199, 20201, 20202, 20203, 20204, 20205, 20206, 20207, 20299, 20301, 20302, 20303, 20304, 20305, 20306, 20307, 20308, 20399, 20401, 20402, 20403, 20404, 20499, 20501, 20502, 20503, 20504, 20505, 20506, 20599, 20601, 20602, 20603, 20604, 20605, 20699, 20701, 20702, 20703, 20704, 20705, 20706, 20707, 20799, 20801, 20802, 20803, 20804, 20899, 20901, 20902, 20903, 20904, 20905, 20906, 20907, 20908, 20999, 21001, 21101, 21102, 21103, 21199, 30101, 30102, 30103, 30104, 30105, 30106, 30107, 30108, 30109, 30110, 30199, 30201, 30202, 30203, 30204, 30205, 30206, 30207, 30208, 30209, 30210, 30211, 30212, 30213, 30214, 30215, 30216, 30217, 30218, 30219, 30220, 30221, 30222, 30223, 30224, 30299, 30301, 30302, 30303, 30304, 30305, 30306, 30307, 30308, 30309, 30310, 30399, 30401, 30402, 30403, 30499, 30501, 30502, 30599, 40101, 40102, 40103, 40104, 40105, 40106, 40107, 40108, 40201, 40301, 40302, 40303, 40304, 40401, 40402, 40501, 40502, 40503, 40504, 40599, 50101, 50102, 50201, 50202, 50203, 50301, 50302, 50303, 50304, 50401, 50402, 50403, 50404, 50501, 50502, 50601, 50602, 50603, 50701, 50702, 50801, 50802, 50803, 50804, 50805, 50901, 50902, 50903, 50904, 50999, 60101, 60102, 60103, 60201, 60202, 60203, 60204, 60301, 60302, 60303, 60304, 60305, 60401, 60402, 60403, 60404, 60405, 60406, 60407, 60408, 60409, 60410, 60501, 60502, 60503, 60599]