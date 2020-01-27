from .. import ChargingNetwork
from .. current import Current
from ...models.evse import get_evse_by_type


def four_station(basic_evse=False, voltage=208, transformer_cap=150, network_type=ChargingNetwork):
    """ Simple, 1 constraint, 1 phase charging network

    Args:
        basic_evse (bool): If True use BASIC EVSE type instead of actual AeroViroment and ClipperCreek types.
        voltage (float): Default voltage at the EVSEs. Does not affect the current rating of the transformer which is
            based on nominal voltages in the network. 277V LL for delta primary and 120V LN for wye secondary. [V]
        network_type (ChargingNetwork like): Type to be returned. Should be ChargingNetwork or a subclass. 

    Returns:
        ChargingNetwork: A ChargingNetwork-like object configured with the EVSEs and constraints of the Caltech ACN.
    """
    network = network_type()

    if basic_evse:
        evse_type = {'AV': 'BASIC', 'CC': 'BASIC'}
    else:
        evse_type = {'AV': 'AeroVironment', 'CC': 'ClipperCreek'}

    # EVSEs
    station_ids = ['SA-001', 'SA-002', 'SA-003', 'SA-004']
    for evse_id in station_ids:
        network.register_evse(get_evse_by_type(evse_id, evse_type['AV']), voltage, 0)

    # Add Simple Constraint
    single_line = Current(station_ids)

    network.add_constraint(single_line, 80, name='CC Pod')

    return network

def one_station(basic_evse=True, voltage=208, transformer_cap=150, network_type=ChargingNetwork):
    """ Simple, 1 constraint, 1 phase charging network

    Args:
        basic_evse (bool): If True use BASIC EVSE type instead of actual AeroViroment and ClipperCreek types.
        voltage (float): Default voltage at the EVSEs. Does not affect the current rating of the transformer which is
            based on nominal voltages in the network. 277V LL for delta primary and 120V LN for wye secondary. [V]
        network_type (ChargingNetwork like): Type to be returned. Should be ChargingNetwork or a subclass. 

    Returns:
        ChargingNetwork: A ChargingNetwork-like object configured with the EVSEs and constraints of the Caltech ACN.
    """
    network = network_type()

    if basic_evse:
        evse_type = {'AV': 'BASIC', 'CC': 'BASIC'}
    else:
        evse_type = {'AV': 'AeroVironment', 'CC': 'ClipperCreek'}

    # EVSEs
    station_ids = ['SSA-001']
    for evse_id in station_ids:
        network.register_evse(get_evse_by_type(evse_id, evse_type['AV']), voltage, 0)

    # Add Simple Constraint
    single_line = Current(station_ids)

    network.add_constraint(single_line, 20, name='CC Pod')

    return network