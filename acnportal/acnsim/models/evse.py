import warnings
from typing import Optional, Dict, Any, Tuple

import numpy as np
from ..base import BaseSimObj

BASIC = "BASIC"
AV = "AeroVironment"
CC = "ClipperCreek"


def get_evse_by_type(station_id, evse_type):
    """ Factory to produce EVSEs of a given type.

    Args:
        station_id (str): Unique identifier of the EVSE.
        evse_type (str): Type of the EVSE. Currently supports 'BASIC', 'AeroVironment', and 'ClipperCreek'.

    Returns:
        EVSE: an EVSE of the specified type and with the specified id.
    """
    if evse_type == BASIC:
        return EVSE(station_id, max_rate=32)
    elif evse_type == AV:
        allowable_rates = [0]
        allowable_rates.extend(i for i in range(6, 33))
        return FiniteRatesEVSE(station_id, allowable_rates)
    elif evse_type == CC:
        allowable_rates = [0, 8, 16, 24, 32]
        return FiniteRatesEVSE(station_id, allowable_rates)


class InvalidRateError(Exception):
    """ Raised when an invalid pilot signal is passed to an EVSE. """

    pass


class StationOccupiedError(Exception):
    """ Raised when a plugin event is called for an EVSE that already has an EV attached. """


class BaseEVSE(BaseSimObj):
    """ Abstract base class to model Electric Vehicle Supply Equipment
    (charging station). This class is meant to be inherited from to
    implement new EVSEs.

    Subclasses must implement the max_rate, allowable_pilot_signals,
    and _valid_rate methods.

    Attributes:
        _station_id (str): Unique identifier of the EVSE.
        _ev (EV): EV currently connected the the EVSE.
        _current_pilot (float): Pilot signal for the current time step.
            [acnsim units]
        is_continuous (bool): If True, this EVSE accepts a continuous
            range of pilot signals. If False, this EVSE accepts only
            a discrete set of pilot signals.
    """

    def __init__(self, station_id):
        """ Initialize a BaseEVSE instance.

        Args:
            station_id (str): Unique identifier of the EVSE.
        """
        self._station_id = station_id
        self._ev = None
        self._current_pilot = 0
        self.is_continuous = True

    @property
    def station_id(self):
        """ Return unique identifier of the EVSE. (str) """
        return self._station_id

    @property
    def ev(self):
        """ Return EV currently connected the the EVSE. (EV) """
        return self._ev

    @property
    def max_rate(self):
        """ Return maximum charging current allowed by the EVSE. (float) """
        raise NotImplementedError

    @property
    def min_rate(self):
        """ Return minimum charging current allowed by the EVSE. (float) """
        return 0

    @property
    def current_pilot(self):
        """ Return pilot signal for the current time step. (float)"""
        return self._current_pilot

    @property
    def allowable_pilot_signals(self):
        """ Returns the allowable pilot signal levels for this EVSE.

        NOT IMPLEMENTED IN BaseEVSE. This method MUST be implemented in
        all subclasses.

        Returns:
            List[float]: List of acceptable pilot signal values or an
                interval of acceptable pilot signal values.
        """
        raise NotImplementedError

    def set_pilot(self, pilot, voltage, period):
        """ Apply a new pilot signal to the EVSE.

        Before applying the new pilot, this method first checks if the pilot is allowed. If it is not, an
        InvalidRateError is raised. If the rate is valid, it is forwarded on to the attached EV if one is present.
        This method is also where EV charging is triggered. Thus it must be called in every time time period where the
        attached EV should receive charge.

        Args:
            pilot (float): New pilot (control signal) to be sent to the attached EV. [A]
            voltage (float): AC voltage provided to the battery charger. [V]
            period (float): Length of the charging period. [minutes]

        Returns:
            None.

        Raises:
            InvalidRateError: Exception raised when pilot is not allowed by the EVSE.
        """
        if self._valid_rate(pilot):
            self._current_pilot = pilot
            if self._ev is not None:
                self._ev.charge(pilot, voltage, period)
        else:
            raise InvalidRateError(
                f"Pilot {pilot} A is not valid for " f"station {self.station_id}."
            )

    def _valid_rate(self, pilot, atol=1e-3):
        """ Check if pilot is in the valid set.

        NOT IMPLEMENTED IN BaseEVSE. This method MUST be implemented in
        all subclasses.

        Args:
            pilot (float): Proposed pilot signal.
            atol: Absolute tolerance used when determining if a pilot
                belongs to the allowable rates set.

        Returns:
            bool: True if the proposed pilot signal is valid. False
                otherwise.
        """
        raise NotImplementedError

    def plugin(self, ev):
        """ Method to attach an EV to the EVSE.

        Args:
            ev (EV): EV which should be attached to the EVSE.

        Returns:
            None.

        Raises:
            StationOccupiedError: Exception raised when plugin is called by an EV is already attached to the EVSE.
        """
        # assert ev.station_id == self.station_id
        if self.ev is None:
            self._ev = ev
        else:
            raise StationOccupiedError(
                f"Station {self._station_id} is occupied with ev "
                f"{self._ev.session_id}."
            )

    def unplug(self):
        """ Method to remove an EV currently attached to the EVSE.

        Sets ev to None and current_pilot to 0.

        Returns:
            None
        """
        self._ev = None
        self._current_pilot = 0

    def _to_dict(
        self, context_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """ Implements BaseSimObj._to_dict. """
        attribute_dict = {}
        nn_attr_lst = ["_station_id", "_current_pilot", "is_continuous"]
        for attr in nn_attr_lst:
            attribute_dict[attr] = getattr(self, attr)

        if self._ev is not None:
            # noinspection PyProtectedMember
            registry, context_dict = self.ev._to_registry(context_dict=context_dict)
            attribute_dict["_ev"] = registry["id"]
        else:
            attribute_dict["_ev"] = None

        return attribute_dict, context_dict

    @classmethod
    def _from_dict_helper(cls, out_obj, attribute_dict, context_dict, loaded_dict):
        out_obj._current_pilot = attribute_dict["_current_pilot"]
        out_obj.is_continuous = attribute_dict["is_continuous"]

        if attribute_dict["_ev"] is not None:
            # noinspection PyProtectedMember
            ev, loaded_dict = BaseSimObj._build_from_id(
                attribute_dict["_ev"], context_dict, loaded_dict=loaded_dict
            )
        else:
            ev = None
        out_obj._ev = ev
        return out_obj, loaded_dict

    @classmethod
    def _from_dict(
        cls,
        attribute_dict: Dict[str, Any],
        context_dict: Dict[str, Any],
        loaded_dict: Optional[Dict[str, BaseSimObj]] = None,
    ) -> Tuple[BaseSimObj, Dict[str, BaseSimObj]]:
        """ Implements BaseSimObj._from_dict. """
        out_obj = cls(attribute_dict["_station_id"])
        return cls._from_dict_helper(out_obj, attribute_dict, context_dict, loaded_dict)


class EVSE(BaseEVSE):
    # TODO: For this class, is a 0 pilot assumed to be allowed?
    """ This class of EVSE allows for charging in a continuous range
    from min_rate to max_rate.

    Attributes:
        See BaseEVSE attributes.
        _max_rate (float): Maximum charging current allowed by the EVSE.
        _min_rate (float): Minimum charging current allowed by the EVSE.
    """

    def __init__(self, station_id, max_rate=float("inf"), min_rate=0):
        """ Initialize an EVSE instance.

        Args:
            See BaseEVSE __init__() Args.
            max_rate (float): Maximum charging current allowed by the
                EVSE.
            min_rate (float): Minimum charging current allowed by the
                EVSE.
        """
        super().__init__(station_id)
        self._max_rate = max_rate
        self._min_rate = min_rate

    @property
    def max_rate(self):
        """ Return maximum charging current allowed by the EVSE. (float) """
        return self._max_rate

    @property
    def min_rate(self):
        """ Return minimum charging current allowed by the EVSE. (float) """
        return self._min_rate

    @property
    def allowable_pilot_signals(self):
        """ Returns the allowable pilot signal levels for this EVSE.

        Implements abstract method allowable_pilot_signals from
        BaseEVSE.

        Returns:
            list[float]: List of 2 values: the min and max
                acceptable values.
        """
        return [self.min_rate, self.max_rate]

    def _valid_rate(self, pilot, atol=1e-3):
        """ Check if pilot is in the valid set.

        Implements abstract method _valid_rate from BaseEVSE.

        Args:
            pilot (float): Proposed pilot signal.
            atol: Absolute tolerance used when determining if a pilot
                belongs to the allowable rates set.

        Returns:
            bool: True if the proposed pilot signal is valid. False
                otherwise.
        """
        return self.min_rate <= pilot + atol and pilot - atol <= self.max_rate

    def _to_dict(
        self, context_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """ Implements BaseSimObj._to_dict. """
        attribute_dict, context_dict = super()._to_dict(context_dict)
        attribute_dict["_max_rate"] = self._max_rate
        attribute_dict["_min_rate"] = self._min_rate

        return attribute_dict, context_dict

    @classmethod
    def _from_dict(
        cls,
        attribute_dict: Dict[str, Any],
        context_dict: Dict[str, Any],
        loaded_dict: Optional[Dict[str, BaseSimObj]] = None,
    ) -> Tuple[BaseSimObj, Dict[str, BaseSimObj]]:
        """ Implements BaseSimObj._from_dict. """
        out_obj = cls(
            attribute_dict["_station_id"],
            max_rate=attribute_dict["_max_rate"],
            min_rate=attribute_dict["_min_rate"],
        )
        return cls._from_dict_helper(out_obj, attribute_dict, context_dict, loaded_dict)


class DeadbandEVSE(BaseEVSE):
    """ Subclass of BaseEVSE which enforces the J1772 deadband between
    0 - 6 A.

    Attributes:
        See BaseEVSE attributes.
        _max_rate (float): Maximum charging current allowed by the EVSE.
        _deadband_end (float): Upper end of the deadband. Pilot signals
            between 0 and this number are not allowed for this EVSE.

    """

    def __init__(
        self, station_id, deadband_end=6, max_rate=float("inf"), min_rate=None
    ):
        """ Initialize a DeadbandEVSE instance.

        Args:
            See EVSE __init__() Args.
            max_rate (float): Maximum charging current allowed by the
                EVSE.
            deadband_end (float): Upper end of the deadband. Pilot
                signals between 0 and this number are not allowed for
                this EVSE.
        """
        super().__init__(station_id)
        self._max_rate = max_rate
        self._deadband_end = deadband_end
        if min_rate is not None:
            warnings.warn(
                f"Keyword argument 'min_rate' is deprecated for class "
                f"DeadbandEVSE. Providing 'min_rate' will raise an "
                f"error in a future release of acnportal.",
                DeprecationWarning,
            )

    @property
    def max_rate(self):
        """ Return maximum charging current allowed by the EVSE. (float) """
        return self._max_rate

    @property
    def deadband_end(self):
        """ Return deadband end of the EVSE. (float) """
        return self._deadband_end

    @property
    def allowable_pilot_signals(self):
        """ Returns the allowable pilot signal levels for this EVSE.

        It is implied that a 0 A signal is allowed.

        Implements abstract method allowable_pilot_signals from
        BaseEVSE.

        Returns:
            list[float]: List of 2 values: the min and max
                acceptable values.
        """
        return [self._deadband_end, self.max_rate]

    def _valid_rate(self, pilot, atol=1e-3):
        """ Check if pilot is in the valid set.

        Overrides super class method. Disallows rates between
        0 - 6 A as per the J1772 standard.

        Args:
            pilot: Proposed pilot signal.
            atol: Absolute tolerance used when determining if a pilot
                belongs to the allowable rates set.

        Returns:
            bool: True if the proposed pilot signal is valid. False
                otherwise.
        """
        return np.isclose(pilot, 0, atol=atol, rtol=0) or (
            self._deadband_end <= pilot + atol and pilot - atol <= self.max_rate
        )

    def _to_dict(
        self, context_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """ Implements BaseSimObj._to_dict. """
        attribute_dict, context_dict = super()._to_dict(context_dict)
        attribute_dict["_max_rate"] = self._max_rate
        attribute_dict["_deadband_end"] = self._deadband_end
        return attribute_dict, context_dict

    @classmethod
    def _from_dict(
        cls,
        attribute_dict: Dict[str, Any],
        context_dict: Dict[str, Any],
        loaded_dict: Optional[Dict[str, BaseSimObj]] = None,
    ) -> Tuple[BaseSimObj, Dict[str, BaseSimObj]]:
        """ Implements BaseSimObj._from_dict. """
        out_obj = cls(
            attribute_dict["_station_id"],
            deadband_end=attribute_dict["_deadband_end"],
            max_rate=attribute_dict["_max_rate"],
        )
        return cls._from_dict_helper(out_obj, attribute_dict, context_dict, loaded_dict)


class FiniteRatesEVSE(BaseEVSE):
    """ Subclass of EVSE which allows for finite allowed rate sets.

    Most functionality remains the same except those differences noted
    below.

    Attributes:
        See BaseEVSE attributes.
        allowable_rates (iterable): Iterable of rates which are allowed
            by the EVSE. On initialization, allowable_rates is converted
            into a list of rates in increasing order that includes 0 and
            contains no duplicate values.

    """

    def __init__(self, station_id, allowable_rates):
        """ Initialize a DeadbandEVSE instance.

        Args:
            See EVSE __init__() Args.
            allowable_rates (iterable): Iterable of rates which are
                allowed by the EVSE. On initialization, allowable_rates
                is converted into a list of rates in increasing order
                that includes 0 and contains no duplicate values.
        """
        super().__init__(station_id)
        allowable_rates = set(allowable_rates)
        allowable_rates.add(0)
        self.allowable_rates = sorted(list(allowable_rates))
        self.is_continuous = False

    @property
    def max_rate(self):
        """ Return maximum charging current allowed by the EVSE. (float) """
        return max(self.allowable_rates)

    @property
    def min_rate(self):
        """ Return minimum charging current allowed by the EVSE. (float) """
        allowable_gt_zero = [r for r in self.allowable_rates if r > 0]
        if len(allowable_gt_zero) > 0:
            return min(allowable_gt_zero)
        else:
            return 0

    @property
    def allowable_pilot_signals(self):
        """ Returns the allowable pilot signal levels for this EVSE.

        Implements abstract method allowable_pilot_signals from
        BaseEVSE.

        Returns:
            list[float]: List of allowable pilot signals.
        """
        return self.allowable_rates

    def _valid_rate(self, pilot, atol=1e-3):
        """ Check if pilot is in the valid set.

        Overrides super class method. Checks if pilot is close to being
        in the allowable set.

        Args:
            pilot: Proposed pilot signal.
            atol: Absolute tolerance used when determining if a pilot
                belongs to the allowable rates set.

        Returns:
            bool: True if the proposed pilot signal is valid. False
                otherwise.
        """
        return np.any(np.isclose(pilot, self.allowable_rates, atol=1e-3, rtol=0))

    def _to_dict(
        self, context_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """ Implements BaseSimObj._to_dict. """
        attribute_dict, context_dict = super()._to_dict(context_dict)
        attribute_dict["allowable_rates"] = self.allowable_rates
        return attribute_dict, context_dict

    @classmethod
    def _from_dict(
        cls,
        attribute_dict: Dict[str, Any],
        context_dict: Dict[str, Any],
        loaded_dict: Optional[Dict[str, BaseSimObj]] = None,
    ) -> Tuple[BaseSimObj, Dict[str, BaseSimObj]]:
        """ Implements BaseSimObj._from_dict. """
        out_obj = cls(attribute_dict["_station_id"], attribute_dict["allowable_rates"])
        return cls._from_dict_helper(out_obj, attribute_dict, context_dict, loaded_dict)
