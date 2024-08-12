import numpy as np
from typing import Optional
from .type_aliases import FloatOrFloatSequence

# Typechecks
def is_bool(x):
    return isinstance(x, bool)

def is_int(x):
    return isinstance(x, int)

def is_float(x):
    return isinstance(x, float)

def is_positive_int(x):
    return is_int(x) and x > 0

def is_positive_float(x):
    return is_float(x) and x > 0

def is_nonnegative_int(x):
    return is_int(x) and x >= 0

def is_power_of_two(n):
    if is_positive_int(n):
        return not n & (n - 1)
    else:
        return False
    
def is_valid_vector(
        x: np.ndarray, 
        bound: Optional[FloatOrFloatSequence] = 1e-3
) -> None:
    """
    Simply check if vector is crazy big or has nans
    currently bounds L^infinity norm. Can also make more sophisticated
    """
    has_nans = np.any(np.isnan(x))
    if has_nans:
        print("Invalid vector, nans detected")
        return False

    within_bounds = np.all(x < bound)
    if not within_bounds:
        print("NOT WITHIN BOUNDS: ", x)
        return False

    return True