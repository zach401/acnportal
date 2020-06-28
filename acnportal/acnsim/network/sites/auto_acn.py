from .. import ChargingNetwork
from ..current import Current
from ...models.evse import get_evse_by_type


def simple_acn(
    station_ids,
    evse_type="BASIC",
    voltage=208,
    aggregate_cap=150,
    network_type=ChargingNetwork,
):
    """ Create a simple, single-phase network with a single aggregate constraint.

    Args:
        station_ids (List[str]): List of station_ids to include in network.
        evse_type (str): Identifier for the EVSE to use. See get_evse_by_type.
        voltage (float): Default voltage at the EVSEs. [V]
        aggregate_cap (float): Limit aggregate power draw to aggregate_cap. Default: 150. [kW]
        network_type (ChargingNetwork like): Type to be returned. Should be ChargingNetwork or a subclass.

    Returns:
        ChargingNetwork: A ChargingNetwork-like object configured with the EVSEs and constraints of the Caltech ACN.
    """
    network = network_type()
    for evse_id in station_ids:
        network.register_evse(get_evse_by_type(evse_id, evse_type), voltage, 0)
    agg = Current(station_ids)

    # Build constraint set
    current_cap = (aggregate_cap / voltage) * 1000
    network.add_constraint(agg, current_cap, name="Aggregate Current")

    return network
