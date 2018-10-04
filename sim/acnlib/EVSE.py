import numpy as np


def get_EVSE_by_type(station_id, evse_type):
    if evse_type == 'EVSE':
        return EVSE(station_id)
    elif evse_type == 'AeroVironment':
        allowable_rates = [0]
        allowable_rates.extend(i for i in range(6, 33))
        return FiniteRatesEVSE(station_id, allowable_rates)
    elif evse_type == 'ClipperCreek':
        allowable_rates = [0, 8, 16, 24, 32]
        return FiniteRatesEVSE(station_id, allowable_rates)


class EVSE:
    '''
    This class models a Electrcial Vehicle Supply Equipment (charging station). Different manufacturers
    have different behavior of their charging stations which is taken into account in this class.

    :ivar string station_id: The station ID for the EVSE.
    :ivar string manufacturer: The manufacturer name of the EVSE. Determines which pilot signal levels that are allowed.
    :ivar float last_applied_pilot_signal: The pilot signal that was applied last iteration.
    :ivar string last_session_id: The ID of the charging session that was using this EVSE last.
    :ivar list(int) allowable_rates: The pilot signals that are allowed for this EVSE.
    '''

    def __init__(self, station_id, max_rate=float('inf'), min_rate=0):
        self.station_id = station_id
        self.last_applied_pilot_signal = 0
        self.last_session_id = None
        self.max_rate = max_rate
        self.min_rate = min_rate

    def change_pilot_signal(self, new_pilot_signal, session_id):
        '''
        Applies a new pilot signal to this EVSE. Also check if the new pilot signal is allowed.

        :param float new_pilot_signal: The new pilot signal that is applied
        :param int session_id: The charging session ID
        '''
        self.last_applied_pilot_signal = new_pilot_signal
        self.last_session_id = session_id
        return self.valid_rate(new_pilot_signal)

    def valid_rate(self, rate):
        return self.min_rate <= rate <= self.max_rate


class FiniteRatesEVSE(EVSE):
    def __init__(self, station_id, allowable_rates):
        super().__init__(station_id, max(allowable_rates), min(allowable_rates))
        self.allowable_rates = allowable_rates

    def valid_rate(self, rate):
        return np.any(np.isclose(rate, self.allowable_rates))