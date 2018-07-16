from EVSE import EVSE
class Garage:

    def __init__(self):
        self.EVSEs = []
        self.test_case = None

        self.define_garage()
        pass

    def define_garage(self):
        self.EVSEs.append(EVSE('CA-148', 'AeroVironment'))
        self.EVSEs.append(EVSE('CA-149', 'AeroVironment'))
        self.EVSEs.append(EVSE('CA-212', 'AeroVironment'))
        self.EVSEs.append(EVSE('CA-213', 'AeroVironment'))
        for i in range(303, 328):
            # CA-303 to CA-327
            if i >= 320 and i <= 323:
                self.EVSEs.append(EVSE('CA-' + str(i), 'ClipperCreek'))
            else:
                self.EVSEs.append(EVSE('CA-' + str(i), 'AeroVironment'))
        for i in range(489, 513):
            # CA-489 to CA-513
            if i >= 493 and i <= 496:
                self.EVSEs.append(EVSE('CA-' + str(i), 'ClipperCreek'))
            else:
                self.EVSEs.append(EVSE('CA-' + str(i), 'AeroVironment'))

    def set_test_case(self, test_case):
        self.test_case = test_case

    def generate_test_case(self):
        return

    def update_state(self, pilot_signals, iteration):
        self.test_case.step(pilot_signals, iteration)

    def event_occured(self, iteration):
        return self.test_case.event_occured(iteration)

    def get_charging_data(self):
        return self.test_case.get_charging_data()

    def get_active_EVs(self, iteration):
        return self.test_case.get_active_EVs(iteration)

    def get_allowable_rates(self, station_id):
        EVSE = next((x for x in self.EVSEs if x.station_id == station_id), None)
        return EVSE.allowable_pilot_signals

    @property
    def last_departure(self):
        return self.test_case.last_departure

    @property
    def max_rate(self):
        return self.test_case.DEFAULT_MAX_RATE

    @property
    def allowable_rates(self):
        return self.test_case.ALLOWABLE_RATES