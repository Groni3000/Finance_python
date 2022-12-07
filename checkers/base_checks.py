import numpy as np, pandas as pd
from typing import Callable, Sequence


class BaseCheckers:
    #? I use *args where it should not be cuz I use generalization in some methods and I don't really know how many args I will provide to 'function' I call.
    @staticmethod
    def is_positive(value: int|float, *args):
        return value > 0
    
    @staticmethod
    def is_negative(value: int|float, *args):
        return value < 0
    
    @staticmethod
    def is_equal(value, equal_to, *args):
        return value == equal_to
    
    @staticmethod
    def check_all(values:Sequence, apply_function:Callable, *func_args):
        return all(apply_function(value, *func_args) for value in values)
        
    @staticmethod
    def check_any(values:Sequence, apply_function:Callable, *func_args):
        return any(apply_function(value, *func_args) for value in values)
        

class SeriesCheckers(BaseCheckers):
    
    @staticmethod
    def same_shapes(series_storage:Sequence[pd.Series|pd.DataFrame]):
        return SeriesCheckers.check_all([obj.shape for obj in series_storage[1:]], SeriesCheckers.is_equal, series_storage[0].shape)