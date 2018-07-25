import math

class EVSE:
    '''
    :ivar string station_id: The station ID for the EVSE.
    :ivar string manufacturer: The manufacturer name of the EVSE. Determines which pilot signal levels that are allowed.
    :ivar float last_applied_pilot_signal: The pilot signal that was applied last iteration.
    :ivar string last_session_id: The ID of the charging session that was using this EVSE last.
    :ivar list(int) allowable_pilot_signals: The pilot signals that are allowed for this EVSE.
    '''

    def __init__(self, station_id, manufacturer):
        self.station_id = station_id
        self.manufacturer = manufacturer
        self.last_applied_pilot_signal = 0
        self.last_session_id = None
        self.allowable_pilot_signals = []
        if manufacturer == "ClipperCreek":
            self.allowable_pilot_signals = [0, 8, 16, 24, 32]
        elif manufacturer == "AeroVironment":
            self.allowable_pilot_signals.append(0)
            for i in range(6, 33):
                self.allowable_pilot_signals.append(i)

    def change_pilot_signal(self, new_pilot_signal, session_id):
        change_ok = True
        new_index = self.allowable_pilot_signals.index(new_pilot_signal)
        old_index = self.allowable_pilot_signals.index(self.last_applied_pilot_signal)
        if math.fabs(new_index - old_index) > 1:
            change_ok = False
        self.last_applied_pilot_signal = new_pilot_signal
        self.last_session_id = session_id
        return change_ok

