import warnings
from typing import Union, Dict, SupportsFloat, List

import pandas as pd


class Current(pd.Series):
    """ A simple representation of currents as an extension of pandas Series.
    Includes addition, subtraction, and multiplication (by scalar) operators.

    Attributes:
        loads (Dict[str, number]): Dictionary which maps a load_id to its coefficient in
        the aggregate current.

    Args:
        loads (Dict[str, number], str, or List[str], pd.Series): If dict, a dictionary
            mapping load_ids to coefficients. If str a load_id. If list, a list of
            load_ids. Default None. If None, loads will begin as an empty dict.
    """

    def __init__(
        self,
        loads: Union[Dict[str, SupportsFloat], str, List[str], pd.Series] = None,
        **kwargs
    ):
        # This subclass doesn't use the data keyword argument, so if the user
        # provides it, a warning is raised.
        if "data" in kwargs and kwargs["data"] is not None:
            warnings.warn(
                "Current class does not use the data keyword of Series constructor. "
                "Use the loads keyword instead."
            )

        # Remove "data" keyword for passing up to constructor.
        if "data" in kwargs:
            del kwargs["data"]

        # Backwards compatibility with previous Current specification methods
        if isinstance(loads, dict):
            super().__init__(loads, **kwargs)
        elif isinstance(loads, str):
            super().__init__({loads: 1}, **kwargs)
        elif loads is not None and all(isinstance(load, str) for load in loads):
            super().__init__({load_id: 1 for load_id in loads}, **kwargs)
        elif isinstance(loads, pd.Series):
            super().__init__(loads, **kwargs)
        elif loads is None:
            # The object type is specified explicitly here to address a
            # warning in pandas 1.x.
            if "dtype" not in kwargs:
                kwargs["dtype"] = "float64"
            super().__init__(**kwargs)
        else:
            raise TypeError(
                "Variable loads should be of type dict, str, pd.Series, or Lst[str]."
            )

    def __add__(self, other):
        """ Return new Current which is the sum of self and other.

        See add_current for description. Major difference is that __add__ returns a new Current object.

        Returns:
            Current: self + other
        Raises:
            TypeError: Raised if other is not of type Current.
        """
        if isinstance(other, Current):
            return Current(self.add(other, fill_value=0))
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
        return Current(self.add(-1 * other, fill_value=0))
