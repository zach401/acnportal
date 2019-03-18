from models.evse import get_EVSE_by_type


class StationOccupiedError(Exception):
    pass


class ChargingNetwork:
    '''
    The ChargingNetwork class describes the infrastructure of the charging network with
    information about the types of the charging station_schedule.
    '''

    def __init__(self):
        self._EVSEs = {}
        pass

    def register_evse(self, evse):
        self._EVSEs[evse.station_id] = evse

    def plugin(self, ev, station_id):
        if station_id in self._EVSEs:
            self._EVSEs[station_id].plugin(ev)
        else:
            raise KeyError('Station {0} not found.'.format(station_id))

    def unplug(self, station_id):
        if station_id in self._EVSEs:
            self._EVSEs[station_id].unplug()
        else:
            raise KeyError('Station {0} not found.'.format(station_id))

    def get_ev(self, station_id):
        if station_id in self._EVSEs:
            return self._EVSEs[station_id].EV
        else:
            raise KeyError('Station {0} not found.'.format(station_id))

    def active_evs(self):
        return [evse.EV for evse in self._EVSEs.values() if evse.EV is not None and not evse.EV.fully_charged]

    def active_station_ids(self):
        return [evse.station_id for evse in self._EVSEs.values() if evse.EV is not None and not evse.EV.fully_charged]

    def update_pilots(self, pilots, i):
        for station_id in self._EVSEs:
            if station_id in pilots and i < len(pilots[station_id]):
                new_rate = pilots[station_id][i]
            else:
                new_rate = 0
            self._EVSEs[station_id].set_pilot(new_rate)

    def get_current_charging_rates(self):
        current_rates = {}
        for station_id, evse in self._EVSEs.items():
            if evse.EV is not None:
                current_rates[station_id] = evse.EV.current_charging_rate
            else:
                current_rates[station_id] = 0
        return current_rates

    def get_space_ids(self):
        return list(self._EVSEs.keys())


def get_caltech_acn(basic=False):
    '''

    :param basic:
    :return:
    '''
    if basic:
        evse_type = {'AV': 'BASIC', 'CC': 'BASIC'}
    else:
        evse_type = {'AV': 'AeroVironment', 'CC': 'ClipperCreek'}

    cn = ChargingNetwork()
    cn.register_evse(get_EVSE_by_type('CA-148', evse_type['AV']))
    cn.register_evse(get_EVSE_by_type('CA-148', evse_type['AV']))
    cn.register_evse(get_EVSE_by_type('CA-149', evse_type['AV']))
    cn.register_evse(get_EVSE_by_type('CA-212', evse_type['AV']))
    cn.register_evse(get_EVSE_by_type('CA-213', evse_type['AV']))
    for i in range(303, 328):
        if i >= 320 or i <= 323:
            cn.register_evse(get_EVSE_by_type('CA-' + str(i), evse_type['CC']))
        else:
            cn.register_evse(get_EVSE_by_type('CA-' + str(i), evse_type['AV']))
    for i in range(489, 514):
        if i >= 493 or i <= 496:
            cn.register_evse(get_EVSE_by_type('CA-' + str(i), evse_type['CC']))
        else:
            cn.register_evse(get_EVSE_by_type('CA-' + str(i), evse_type['AV']))
    return cn


# def get_caltech_acn_basic():
#     '''
#     Creates the _EVSEs of the garage.
#
#     :return: None
#     '''
#     cn = ChargingNetwork()
#     cn.register_evse(get_EVSE_by_type('CA-148', 'BASIC'))
#     cn.register_evse(get_EVSE_by_type('CA-148', 'BASIC'))
#     cn.register_evse(get_EVSE_by_type('CA-149', 'BASIC'))
#     cn.register_evse(get_EVSE_by_type('CA-212', 'BASIC'))
#     cn.register_evse(get_EVSE_by_type('CA-213', 'BASIC'))
#     for i in range(303, 328):
#         # CA-303 to CA-327
#         if i >= 320 and i <= 323:
#             cn.register_evse(get_EVSE_by_type('CA-' + str(i), 'BASIC'))
#         else:
#             cn.register_evse(get_EVSE_by_type('CA-' + str(i), 'BASIC'))
#     for i in range(489, 514):
#         # CA-489 to CA-513
#         if i >= 493 and i <= 496:
#             cn.register_evse(get_EVSE_by_type('CA-' + str(i), 'BASIC'))
#         else:
#             cn.register_evse(get_EVSE_by_type('CA-' + str(i), 'BASIC'))
#     return cn