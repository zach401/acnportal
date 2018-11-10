import numpy as np

BASIC = 'BASIC'
AV = 'AeroVironment'
CC = 'ClipperCreek'

def get_EVSE_by_type(station_id, evse_type):
    '''
    Factory to produce _EVSEs of a given type.

    :param station_id: unique identifier of the station.
    :param evse_type:
    :return:
    '''
    if evse_type == BASIC:
        return EVSE(station_id)
    elif evse_type == AV:
        allowable_rates = [0]
        allowable_rates.extend(i for i in range(6, 33))
        return FiniteRatesEVSE(station_id, allowable_rates)
    elif evse_type == CC:
        allowable_rates = [0, 8, 16, 24, 32]
        return FiniteRatesEVSE(station_id, allowable_rates)


class InvalidRateError(Exception):
    pass


class Empty(Exception):
    pass


class StationOccupiuedError(Exception):
    pass


class EVSE:
    '''
    Base class to  model Electric Vehicle Supply Equipment (charging station).
    This model allows for charging in a continuous range from min_rate to max_rate.

    :ivar string station_id: The station ID for the EVSE.
    :ivar string manufacturer: The manufacturer name of the EVSE. Determines which pilot signal levels that are allowed.
    :ivar float last_applied_pilot_signal: The pilot signal that was applied last iteration.
    :ivar string last_session_id: The ID of the charging session that was using this EVSE last.
    :ivar float max_rate: Maximum allowed charging rate of the EVSE.
    :ivar float min_rate: Minimum allowed charging rate of the EVSE.
    '''

    def __init__(self, station_id, max_rate=float('inf'), min_rate=0):
        self._station_id = station_id
        self._EV = None
        self._max_rate = max_rate
        self._min_rate = min_rate
        self._current_pilot = 0

    @property
    def station_id(self):
        return self._station_id

    @property
    def EV(self):
        return self._EV

    @property
    def max_rate(self):
        return self._max_rate

    @property
    def min_rate(self):
        return self._min_rate

    @property
    def current_pilot(self):
        return self._current_pilot

    def set_pilot(self, pilot):
        '''
        Applies a new pilot signal to this EVSE. Also check if the new pilot signal is allowed.

        :param float new_pilot_signal: The new pilot signal that is applied

        :raises Empty: raises exception when no EV is assigned to the EVSE.
        :raises InvalidRateError: raises error when assigned pilot is not a valid rate.
        '''
        if self._valid_rate(pilot):
            self._current_pilot = pilot
            if self._EV is not None:
                self._EV.charge(pilot)
        else:
            raise InvalidRateError('Pilot {0} A is not valid for for station {1}'.format(pilot, self.station_id))

    def _valid_rate(self, rate):
        return self.min_rate <= rate <= self.max_rate

    def plugin(self, ev):
        if self.EV is None:
            self._EV = ev
        else:
            raise StationOccupiuedError('Station {0} is occupied with EV {1}'.format(self._station_id,
                                                                                     self._EV.session_id))

    def unplug(self):
        self._EV = None
        self._current_pilot = 0


class FiniteRatesEVSE(EVSE):
    '''
    Child class to  model Electric Vehicle Supply Equipment which only allow finite sets of charging rates.

    :ivar string station_id: The station ID for the EVSE.
    :ivar string manufacturer: The manufacturer name of the EVSE. Determines which pilot signal levels that are allowed.
    :ivar float last_applied_pilot_signal: The pilot signal that was applied last iteration.
    :ivar string last_session_id: The ID of the charging session that was using this EVSE last.
    :ivar float max_rate: Maximum allowed charging rate of the EVSE.
    :ivar float min_rate: Minimum allowed charging rate of the EVSE.
    :ivar list(int) allowable_rates: The pilot signals that are allowed for this EVSE.
    '''
    def __init__(self, station_id, allowable_rates):
        super().__init__(station_id, max(allowable_rates), min(allowable_rates))
        self.allowable_rates = allowable_rates

    def _valid_rate(self, rate):
        return np.any(np.isclose(rate, self.allowable_rates, atol=1e-3))

    def set_pilot(self, pilot):
        _pilot = np.round(pilot)
        super().set_pilot(_pilot)