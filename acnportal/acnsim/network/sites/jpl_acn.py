from .. import ChargingNetwork
from ..current import Current
from ...models.evse import get_evse_by_type
import numpy as np


def _register_evses(network, evse_ids, evse_type, voltage, angle):
    for evse_id in evse_ids:
        network.register_evse(get_evse_by_type(evse_id, evse_type), voltage, angle)


def _add_line2line_evses(
    network, ab, bc, ca, voltage, evse_type, phi_ab=30, phi_bc=-90, phi_ca=150
):
    _register_evses(network, ab, evse_type, voltage, phi_ab)
    _register_evses(network, bc, evse_type, voltage, phi_bc)
    _register_evses(network, ca, evse_type, voltage, phi_ca)

    currents = {}
    currents["ab"] = Current(ab)
    currents["bc"] = Current(bc)
    currents["ca"] = Current(ca)

    currents["a"] = currents["ab"] - currents["ca"]
    currents["b"] = currents["bc"] - currents["ab"]
    currents["c"] = currents["ca"] - currents["bc"]

    return currents


def jpl_acn(
    basic_evse=False,
    voltage=208,
    first_transformer_cap=45,
    third_fourth_transformer_cap=150,
    network_type=ChargingNetwork,
):
    """ Predefined ChargingNetwork for the JPL ACN.

    Args:
        basic_evse (bool): If True use BASIC EVSE type instead of actual AeroViroment types.
        voltage (float): Default voltage at the EVSEs. Does not affect the current rating of the transformer which is
            based on nominal voltages in the network. 277V LL for delta primary and 120V LN for wye secondary. [V]
        first_transformer_cap (float): Capacity of transformer feeding the 1st floor. Default: 45. [kW]
        third_fourth_transformer_cap (float): Capacity of transformer feeding the 3rd and 4th floors. Default: 150. [kW]
        network_type (ChargingNetwork like): Type to be returned. Should be ChargingNetwork or a subclass.

    Returns:
        ChargingNetwork: A ChargingNetwork-like object configured with the EVSEs and constraints of the JPL ACN.
    """
    network = network_type()

    if basic_evse:
        evse_type = "BASIC"
    else:
        evse_type = "AeroVironment"

    def _delta_wye_transformer(name, Isec, cap, secondary_voltage=120, n=4):
        Ipa = (1 / n) * (Isec["a"] - Isec["c"])
        Ipb = (1 / n) * (Isec["b"] - Isec["a"])
        Ipc = (1 / n) * (Isec["c"] - Isec["b"])

        primary_side_constr = cap * 1000 / 3 / secondary_voltage * np.sqrt(3)
        secondary_side_constr = cap * 1000 / 3 / secondary_voltage
        network.add_constraint(
            Isec["a"], secondary_side_constr, name="{0} Secondary A".format(name)
        )
        network.add_constraint(
            Isec["b"], secondary_side_constr, name="{0} Secondary B".format(name)
        )
        network.add_constraint(
            Isec["c"], secondary_side_constr, name="{0} Secondary C".format(name)
        )
        network.add_constraint(
            Ipa, primary_side_constr, name="{0} Primary A".format(name)
        )
        network.add_constraint(
            Ipb, primary_side_constr, name="{0} Primary B".format(name)
        )
        network.add_constraint(
            Ipc, primary_side_constr, name="{0} Primary C".format(name)
        )

    # -------- 1st Floor 45 kW Transformer -----------------
    # Sub-panel 1 (Max 100 A / phase)
    first_floor_sp1 = _add_line2line_evses(
        network, ["AG-1F12", "AG-1F14"], [], ["AG-1F11", "AG-1F13"], voltage, evse_type
    )

    # Sub-panel 2 (Max 100 A / phase)
    first_floor_sp2 = _add_line2line_evses(
        network,
        ["AG-1F03", "AG-1F06"],
        ["AG-1F01", "AG-1F04"],
        ["AG-1F02", "AG-1F05"],
        voltage,
        evse_type,
    )

    # Additional EVSEs on main panel
    add_first_floor = _add_line2line_evses(
        network, ["AG-1F10"], ["AG-1F07", "AG-1F09"], ["AG-1F08"], voltage, evse_type
    )
    first_floor_transformer = dict()
    for p in "abc":
        first_floor_transformer[p] = (
            add_first_floor[p] + first_floor_sp1[p] + first_floor_sp2[p]
        )

    # -------- 3rd and 4th Floors 150 kW Transformer -----------------

    # 3rd Floor (Max 225 A / phase)
    third_floor_panel = _add_line2line_evses(
        network,
        [
            "AG-3F16",
            "AG-3F17",
            "AG-3F20",
            "AG-3F23",
            "AG-3F25",
            "AG-3F26",
            "AG-3F29",
            "AG-3F33",
        ],
        ["AG-3F18", "AG-3F21", "AG-3F27", "AG-3F30", "AG-3F31"],
        ["AG-3F15", "AG-3F19", "AG-3F22", "AG-3F24", "AG-3F28", "AG-3F32"],
        voltage,
        evse_type,
    )

    # 4th Floor (Max 225 A / phase)
    fourth_floor_panel = _add_line2line_evses(
        network,
        [
            "AG-4F35",
            "AG-4F36",
            "AG-4F39",
            "AG-4F42",
            "AG-4F44",
            "AG-4F45",
            "AG-4F48",
            "AG-4F52",
        ],
        ["AG-4F37", "AG-4F40", "AG-4F46", "AG-4F49", "AG-4F50"],
        ["AG-4F34", "AG-4F38", "AG-4F41", "AG-4F43", "AG-4F47", "AG-4F51"],
        voltage,
        evse_type,
    )
    third_fourth_transformer = {
        p: third_floor_panel[p] + fourth_floor_panel[p] for p in "abc"
    }

    # ------ Add Constraints -----------------
    # Line Constraints
    for p in "abc":
        network.add_constraint(
            first_floor_sp1[p], 100, name="First Floor SP1 I_{0}".format(p)
        )
        network.add_constraint(
            first_floor_sp2[p], 100, name="First Floor SP2 I_{0}".format(p)
        )
        network.add_constraint(
            third_floor_panel[p], 225, name="Third Floor Panel I_{0}".format(p)
        )
        network.add_constraint(
            fourth_floor_panel[p], 225, name="Fourth Floor Panel I_{0}".format(p)
        )

    # Transformer Constraints
    _delta_wye_transformer(
        "First Floor Transformer", first_floor_transformer, first_transformer_cap
    )
    _delta_wye_transformer(
        "Third/Fourth Floor Transformer",
        third_fourth_transformer,
        third_fourth_transformer_cap,
    )

    return network
