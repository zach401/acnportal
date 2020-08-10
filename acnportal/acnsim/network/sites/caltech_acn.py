from .. import ChargingNetwork
from ..current import Current
from ...models.evse import get_evse_by_type


def caltech_acn(
    basic_evse=False, voltage=208, transformer_cap=150, network_type=ChargingNetwork
):
    """ Predefined ChargingNetwork for the Caltech ACN.

    Args:
        basic_evse (bool): If True use BASIC EVSE type instead of actual AeroViroment and ClipperCreek types.
        voltage (float): Default voltage at the EVSEs. Does not affect the current rating of the transformer which is
            based on nominal voltages in the network. 277V LL for delta primary and 120V LN for wye secondary. [V]
        transformer_cap (float): Capacity of the transformer in the CaltechACN. Default: 150. [kW]
        network_type (ChargingNetwork like): Type to be returned. Should be ChargingNetwork or a subclass.

    Returns:
        ChargingNetwork: A ChargingNetwork-like object configured with the EVSEs and constraints of the Caltech ACN.
    """
    network = network_type()

    if basic_evse:
        evse_type = {"AV": "BASIC", "CC": "BASIC"}
    else:
        evse_type = {"AV": "AeroVironment", "CC": "ClipperCreek"}

    # Define the sets of EVSEs in the Caltech ACN.
    CC_pod_ids = [
        "CA-322",
        "CA-493",
        "CA-496",
        "CA-320",
        "CA-495",
        "CA-321",
        "CA-323",
        "CA-494",
    ]
    AV_pod_ids = [
        "CA-324",
        "CA-325",
        "CA-326",
        "CA-327",
        "CA-489",
        "CA-490",
        "CA-491",
        "CA-492",
    ]
    AB_ids = (
        ["CA-{0}".format(i) for i in [308, 508, 303, 513, 310, 506, 316, 500, 318, 498]]
        + AV_pod_ids
        + CC_pod_ids
    )
    BC_ids = [
        "CA-{0}".format(i)
        for i in [304, 512, 305, 511, 313, 503, 311, 505, 317, 499, 148, 149, 212, 213]
    ]
    CA_ids = [
        "CA-{0}".format(i)
        for i in [307, 509, 309, 507, 306, 510, 315, 501, 319, 497, 312, 504, 314, 502]
    ]

    # Add Caltech EVSEs
    for evse_id in AB_ids:
        if evse_id not in CC_pod_ids:
            network.register_evse(
                get_evse_by_type(evse_id, evse_type["AV"]), voltage, 30
            )
        else:
            network.register_evse(
                get_evse_by_type(evse_id, evse_type["CC"]), voltage, 30
            )
    for evse_id in BC_ids:
        network.register_evse(get_evse_by_type(evse_id, evse_type["AV"]), voltage, -90)
    for evse_id in CA_ids:
        network.register_evse(get_evse_by_type(evse_id, evse_type["AV"]), voltage, 150)

    # Add Caltech Constraint Set
    CC_pod = Current(CC_pod_ids)
    AV_pod = Current(AV_pod_ids)
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
    network.add_constraint(CC_pod, 80, name="CC Pod")
    network.add_constraint(AV_pod, 80, name="AV Pod")
    network.add_constraint(I3a, secondary_side_constr, name="Secondary A")
    network.add_constraint(I3b, secondary_side_constr, name="Secondary B")
    network.add_constraint(I3c, secondary_side_constr, name="Secondary C")
    network.add_constraint(I2a, primary_side_constr, name="Primary A")
    network.add_constraint(I2b, primary_side_constr, name="Primary B")
    network.add_constraint(I2c, primary_side_constr, name="Primary C")

    return network


def CaltechACN(
    basic_evse=False, voltage=208, transformer_cap=150, network_type=ChargingNetwork
):
    """ Wrapper around caltech_acn for backward compatibility. Will be removed in future release. """
    print(
        "WARNING: CaltechACN will be removed in a future release. Please use caltech_acn()."
    )
    return caltech_acn(basic_evse, voltage, transformer_cap, network_type)
