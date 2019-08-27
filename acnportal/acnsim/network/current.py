import cmath
import math
from copy import deepcopy
import numpy as np
import pandas as pd

class Current(pd.Series):
    """ A simple representation of currents as an extension of pandas Series.
    Includes addition, subtraction, and multiplication (by scalar) operators.

    Attributes:
        loads (Dict[str, number]): Dictionary which maps a load_id to its coefficient in the aggregate current.

    Args:
        loads (Dict[str, number], str, or List[str]): If dict, a dictionary mapping load_ids to coefficients. If str a
            load_id. If list, a list of load_ids. Default None. If None, loads will begin as an empty dict.
    """
    # Backwards compatibility with previous Current specification methods
    # TODO: Decide if we want to keep accepting all methods or require
    # string : float dict inputs.
    def __init__(self, loads=None):
        if isinstance(loads, dict):
            super().__init__(loads)
        elif isinstance(loads, str):
            super().__init__({loads : 1})
        elif loads is None:
            super().__init__()
        else:
            super().__init__({load_id: 1 for load_id in loads})

    def __add__(self, other):
        """ Return new Current which is the sum of self and other.

        See add_current for description. Major difference is that __add__ returns a new Current object.

        Returns:
            Current: self + other
        Raises:
            TypeError: Raised if other is not of type Current.
        """
        if isinstance(other, Current):
            return self.add(other, fill_value=0)
        else:
            TypeError("Must be of type Current.")

    # Allow for right addition as well.
    __radd__ = __add__

    def __sub__(self, other):
        """ Return Current which is self minus other.

        Args:
            other (Current): Current to be subtracted from self.
        Returns:
            Current: self - other
        """
        # TODO: Bug: for some reason the line below doesn't work
        # return self + ((-1 * other))
        return self.add(-1 * other, fill_value=0)
