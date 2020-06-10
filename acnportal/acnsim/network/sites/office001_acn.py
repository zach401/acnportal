from .. import ChargingNetwork
from ..current import Current
from ...models.evse import get_evse_by_type


def office001_acn(
    basic_evse=False, voltage=208, transformer_cap=50, network_type=ChargingNetwork
):
    """ Predefined ChargingNetwork for the Office001 ACN.

    Args:
        basic_evse (bool): If True use BASIC EVSE type instead of actual AeroViroment types.
        voltage (float): Default voltage at the EVSEs. Does not affect the current rating of the transformer which is
            based on nominal voltages in the network. 277V LL for delta primary and 120V LN for wye secondary. [V]
        transformer_cap (float): Capacity of the transformer feeding the network. Default: 150. [kW]
        network_type (ChargingNetwork like): Type to be returned. Should be ChargingNetwork or a subclass.

    Returns:
        ChargingNetwork: A ChargingNetwork-like object configured with the EVSEs and constraints of the Office 1 ACN.
    """
    network = network_type()

    if basic_evse:
        evse_type = "BASIC"
    else:
        evse_type = "AeroVironment"

    # Define the sets of EVSEs
    AB_ids = ["01", "04", "07"]
    BC_ids = ["02", "05", "08"]
    CA_ids = ["03", "06"]

    # Add EVSEs
    for evse_id in AB_ids:
        network.register_evse(get_evse_by_type(evse_id, evse_type), voltage, 30)
    for evse_id in BC_ids:
        network.register_evse(get_evse_by_type(evse_id, evse_type), voltage, -90)
    for evse_id in CA_ids:
        network.register_evse(get_evse_by_type(evse_id, evse_type), voltage, 150)

    # Define currents
    AB = Current(AB_ids)
    BC = Current(BC_ids)
    CA = Current(CA_ids)

    # Define intermediate currents
    I3a = AB - CA
    I3b = BC - AB
    I3c = CA - BC
    I2a = (1 / 4) * (I3a - I3c)
    I2b = (1 / 4) * (I3b - I3a)
    I2c = (1 / 4) * (I3c - I3b)

    # Build constraint set
    primary_side_constr = transformer_cap * 1000 / 3 / 277
    secondary_side_constr = transformer_cap * 1000 / 3 / 120
    network.add_constraint(I3a, secondary_side_constr, name="Secondary A")
    network.add_constraint(I3b, secondary_side_constr, name="Secondary B")
    network.add_constraint(I3c, secondary_side_constr, name="Secondary C")
    network.add_constraint(I2a, primary_side_constr, name="Primary A")
    network.add_constraint(I2b, primary_side_constr, name="Primary B")
    network.add_constraint(I2c, primary_side_constr, name="Primary C")

    return network
