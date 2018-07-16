
class EVSE:

    def __init__(self, station_id, manufacturer):
        self.station_id = station_id
        self.manufacturer = manufacturer
        self.allowable_pilot_signals = []
        if manufacturer == "ClipperCreek":
            self.allowable_pilot_signals = [0, 8, 16, 24, 32]
        elif manufacturer == "AeroVironment":
            self.allowable_pilot_signals.append(0)
            for i in range(6, 33):
                self.allowable_pilot_signals.append(i)
