import cmath
import math
from copy import deepcopy


class ConstraintSet:
    """ A collection of current constraints.

    Args:
        constraints (List[Constraint]): A list of constraints which make up the constraint set.

    Attributes:
        See Args.
    """
    def __init__(self, constraints=None):
        self.constraints = constraints if constraints is not None else []

    def add_constraint(self, current, limit, name=None):
        """ Add an additional constraint to the constraint set.

        Args:
            See Constraint.

        Returns:
            None
        """
        if name is None:
            name = '_const_{0}'.format(len(self.constraints))
        self.constraints.append(Constraint(current, limit, name))

    def constraint_current(self, constraint, load_currents, angles, t=0, linear=False):
        """ Return the current subject to the given constraint.

        Args:
            constraint (Constraint): Constraint object describing the current.
            load_currents (Dict[str, List[number]]): Dictionary mapping load_ids to schedules of charging rates.
            angles (Dict[str, float]): Dictionary mapping load_ids to the phase angle of the voltage feeding them.
            t (int): Index into the charging rate schedule where feasibility should be checked.
            linear (bool): If True, linearize all constraints to a more conservative but easier to compute constraint by
                ignoring the phase angle and taking the absolute value of all load coefficients. Default False.

        Returns:
            complex: Current subject to the given constraint.
        """
        acc = 0
        for load_id in constraint.loads:
            if load_id in load_currents:
                if linear:
                    acc += abs(constraint.loads[load_id]) * load_currents[load_id][t]
                else:
                    acc += cmath.rect(constraint.loads[load_id] * load_currents[load_id][t],
                                      math.radians(angles[load_id]))
        return complex(acc)

    def is_feasible(self, load_currents, angles, t=0, linear=False):
        """ Return if a set of current magnitudes for each load are feasible.

        Args:
            load_currents (Dict[str, List[number]]): Dictionary mapping load_ids to schedules of charging rates.
            angles (Dict[str, float]): Dictionary mapping load_ids to the phase angle of the voltage feeding them.
            t (int): Index into the charging rate schedule where feasibility should be checked.
            linear (bool): If True, linearize all constraints to a more conservative but easier to compute constraint by
                ignoring the phase angle and taking the absolute value of all load coefficients. Default False.

        Returns:
            bool: If load_currents is feasible at time t according to this constraint set.
        """
        for constraint in self.constraints:
            mag = self.constraint_current(constraint, load_currents, angles, t, linear)
            if abs(mag) > constraint.limit:
                return False
        return True


class Constraint:
    """ A simple representation of a current constraint.

    Args:
        current (Current): Current whose magnitude is bound by this constraint.
        limit (number): Limit on the magnitude of the current.
        name (str): Identifier of the constraint.

    Attributes:
        See Args.
    """
    def __init__(self, curr, limit, name):
        self.current = curr
        self.limit = limit
        self.name = name

    @property
    def loads(self):
        """ Returns the loads of the underlying current object.

        Returns:
            See Current for description of loads dict.
        """
        return self.current.loads


class Current:
    """ A simple representation of currents include addition, subtraction, and multiplication operators.

    Attributes:
        loads (Dict[str, number]): Dictionary which maps a load_id to its coefficient in the aggregate current.

    Args:
        loads (Dict[str, number], str, or List[str]): If dict, a dictionary mapping load_ids to coefficients. If str a
            load_id. If list, a list of load_ids. Default None. If None, loads will begin as an empty dict.
    """
    def __init__(self, loads=None):
        if isinstance(loads, dict):
            self.loads = loads
        elif isinstance(loads, str):
            self.loads = {loads: 1}
        elif loads is None:
            self.loads = {}
        else:
            self.loads = {load_id: 1 for load_id in loads}

    def add_current(self, curr):
        """ Add Current object to self in place.

        Args:
            curr (Current): Second current which will be added to self.

        Returns:
            None
        """
        for load_id in curr.loads:
            if load_id in self.loads:
                self.loads[load_id] += curr.loads[load_id]
            else:
                self.loads[load_id] = curr.loads[load_id]

    def multiply_by_const(self, const):
        """ Multiply self by a constant.

        Args:
            const (number): Coefficient which the Current should be scaled by.

        Returns:
            None
        """
        for val in self.loads.values():
            val *= const

    def __add__(self, other):
        """ Return new Current which is the sum of self and other.

        See add_current for description. Major difference is that __add__ returns a new Current object.

        Returns:
            Current: self + other
        Raises:
            TypeError: Raised if other is not of type Current.
        """
        new_current = deepcopy(self)
        if isinstance(other, Current):
            new_current.add_current(other)
        else:
            TypeError("Must be of type Current.")
        return new_current

    # Allow for right addition as well.
    __radd__ = __add__

    def __mul__(self, other):
        """ Return new Current which is self multiplied by a constant.

        See multiply_by_const for description. Major difference is that __mul__ returns a new Current object.


        Returns:
            Current: other * self
        """
        new_current = deepcopy(self)
        for load_id in new_current.loads:
            new_current.loads[load_id] *= float(other)
        return new_current

    # Allow for right multiply.
    __rmul__ = __mul__

    def __neg__(self):
        """ Return Current which is the negative of self.

        Returns:
            Current: -1 * self
        """
        return (-1) * self

    def __sub__(self, other):
        """ Return Current which is self minus other.

        Args:
            other (Current): Current to be subtracted from self.
        Returns:
            Current: self - other
        """
        return self + (-1) * other
