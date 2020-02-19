import numpy as np
from .. import base

BASIC = 'BASIC'
AV = 'AeroVironment'
CC = 'ClipperCreek'


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
    pass


class EVSE(base.BaseSimObj):
    """ Class to model Electric Vehicle Supply Equipment (charging station).

    This base class allows for charging in a continuous range from min_rate to max_rate.

    Args:
        station_id (str): Unique identifier of the EVSE.
        ev (EV): EV currently connected the the EVSE.
        max_rate (float): Maximum charging current allowed by the EVSE.
        min_rate (float): Minimum charging current allowed by the EVSE.
        current_pilot (float): Pilot signal for the current time step. [acnsim units]
    """
    def __init__(self, station_id, max_rate=float('inf'), min_rate=0):
        self._station_id = station_id
        self._ev = None
        self._max_rate = max_rate
        self._min_rate = min_rate
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
        return self._max_rate

    @property
    def min_rate(self):
        """ Return minimum charging current allowed by the EVSE. (float) """
        return self._min_rate

    @property
    def current_pilot(self):
        """ Return pilot signal for the current time step. (float)"""
        return self._current_pilot

    @property
    def allowable_pilot_signals(self):
        """ Returns the allowable pilot signal levels for this EVSE.

        Returns:
            list[float]: List of 2 values: the min and max
                acceptable values.
        """
        return [self.min_rate, self.max_rate]

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
            raise InvalidRateError('Pilot {0} A is not valid for for station {1}'.format(pilot, self.station_id))

    def _valid_rate(self, pilot):
        """ Check if pilot is in the valid set.

        Args:
            pilot (float): Proposed pilot signal.

        Returns:
            bool: True if the proposed pilot signal is valid. False otherwise.
        """
        return self.min_rate <= pilot <= self.max_rate

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
            raise StationOccupiedError('Station {0} is occupied with ev {1}'.format(self._station_id,
                                                                                    self._ev.session_id))

    def unplug(self):
        """ Method to remove an EV currently attached to the EVSE.

        Sets ev to None and current_pilot to 0.

        Returns:
            None
        """
        self._ev = None
        self._current_pilot = 0

    def to_dict(self, context_dict=None):
        """ Implements BaseSimObj.to_dict. """
        attribute_dict = {}
        nn_attr_lst = ['_station_id', '_max_rate', '_min_rate',
                       '_current_pilot', 'is_continuous']
        for attr in nn_attr_lst:
            attribute_dict[attr] = getattr(self, attr)

        if self._ev is not None:
            registry, context_dict = self.ev.to_registry(
                context_dict=context_dict)
            attribute_dict['_ev'] = registry['id']
        else:
            attribute_dict['_ev'] = None

        return attribute_dict, context_dict

    @classmethod
    def from_dict(cls, attribute_dict, context_dict,
                  loaded_dict=None, cls_kwargs=None):
        """ Implements BaseSimObj.from_dict. """
        cls_kwargs, = base.none_to_empty_dict(cls_kwargs)
        out_obj = cls(
            attribute_dict['_station_id'],
            max_rate=attribute_dict['_max_rate'],
            min_rate=attribute_dict['_min_rate'],
            **cls_kwargs
        )
        out_obj._current_pilot = attribute_dict['_current_pilot']
        out_obj.is_continuous = attribute_dict['is_continuous']

        if attribute_dict['_ev'] is not None:
            ev, loaded_dict = base.build_from_id(
                attribute_dict['_ev'], context_dict, loaded_dict=loaded_dict)
        else:
            ev = None
        out_obj._ev = ev

        return out_obj, loaded_dict


class DeadbandEVSE(EVSE):
    """ Subclass of EVSE which enforces the J1772 deadband between 0 - 6 A.

    Most functionality remains the same except for a new _valid_rate function.

    """

    def __init__(self, station_id, deadband_end=6, max_rate=float('inf'), min_rate=0):
        super().__init__(station_id, max_rate, min_rate)
        self._deadband_end = deadband_end

    @property
    def allowable_pilot_signals(self):
        """ Returns the allowable pilot signal levels for this EVSE.

        Returns:
            list[float]: List of 2 values: the min and max
                acceptable values.
        """
        return [self._deadband_end, self.max_rate]

    def _valid_rate(self, pilot, atol=1e-3):
        """ Overrides super class method. Disallows rates between 0 - 6 A as per the J1772 standard.

        Args:
            pilot: Proposed pilot signal.
            atol: Absolute tolerance used when determining if a pilot belongs to the allowable rates set.

        Returns:
            bool: True if the proposed pilot signal is valid. False otherwise.
        """
        return np.isclose(pilot, 0, atol) or pilot > self._deadband_end

    def to_dict(self, context_dict=None):
        """ Implements BaseSimObj.to_dict. """
        attribute_dict, context_dict = super().to_dict(context_dict)
        attribute_dict['_deadband_end'] = self._deadband_end
        return attribute_dict, context_dict

    @classmethod
    def from_dict(cls, attribute_dict, context_dict,
                  loaded_dict=None, cls_kwargs=None):
        """ Implements BaseSimObj.from_dict. """
        cls_kwargs, = base.none_to_empty_dict(cls_kwargs)
        cls_kwargs['deadband_end'] = attribute_dict['_deadband_end']
        out_obj, loaded_dict = super().from_dict(
            attribute_dict, context_dict, loaded_dict, cls_kwargs)
        return out_obj, loaded_dict


class FiniteRatesEVSE(EVSE):
    """ Subclass of EVSE which allows for finite allowed rate sets.

    Most functionality remains the same except those differences noted below.

    Attributes:
        allowable_rates (iterable): Iterable of rates which are allowed by the EVSE.
            On initialization, allowable_rates is converted into a list of rates
            in increasing order that includes 0 and contains no duplicate values.

    """
    def __init__(self, station_id, allowable_rates):
        super().__init__(station_id, max(allowable_rates))
        allowable_rates = set(allowable_rates)
        allowable_rates.add(0)
        self.allowable_rates = sorted(list(allowable_rates))
        self.is_continuous = False

    @property
    def allowable_pilot_signals(self):
        """ Returns the allowable pilot signal levels for this EVSE.

        Returns:
            list[float]: List of allowable pilot signals.
        """
        return self.allowable_rates

    def _valid_rate(self, pilot, atol=1e-3):
        """ Overrides super class method. Checks if pilot is close to being in the allowable set.

        Args:
            pilot: Proposed pilot signal.
            atol: Absolute tolerance used when determining if a pilot belongs to the allowable rates set.

        Returns:
            bool: True if the proposed pilot signal is valid. False otherwise.
        """
        return np.any(np.isclose(pilot, self.allowable_rates, atol=1e-3))

    def to_dict(self, context_dict=None):
        """ Implements BaseSimObj.to_dict. """
        attribute_dict, context_dict = super().to_dict(context_dict)
        attribute_dict['allowable_rates'] = self.allowable_rates
        return attribute_dict, context_dict

    @classmethod
    def from_dict(cls, attribute_dict, context_dict,
                  loaded_dict=None, cls_kwargs=None):
        """ Implements BaseSimObj.from_dict. """
        cls_kwargs, = base.none_to_empty_dict(cls_kwargs)
        out_obj = cls(
            attribute_dict['_station_id'],
            attribute_dict['allowable_rates'],
            **cls_kwargs
        )
        out_obj._max_rate = attribute_dict['_max_rate']
        out_obj._min_rate = attribute_dict['_min_rate']
        out_obj._current_pilot = attribute_dict['_current_pilot']
        out_obj.is_continuous = attribute_dict['is_continuous']

        if attribute_dict['_ev'] is not None:
            ev, loaded_dict = base.build_from_id(
                attribute_dict['_ev'], context_dict, loaded_dict=loaded_dict)
        else:
            ev = None
        out_obj._ev = ev

        return out_obj, loaded_dict
