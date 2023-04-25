import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields



@dataclass
class GlobalArgs():
    """
    Global args for this repo
    """
    manual_seed: bool = True
    random_seed: int = 42
    min_words: int = 20