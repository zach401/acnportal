from . import ChargingNetwork
from .constraint_set import Current
from acnportal.acnsim.models.evse import get_evse_by_type
import math


class CaltechACN(ChargingNetwork):
    """ Predefined ChargingNetwork for the Caltech ACN.

    Args:
        basic_evse (bool): If True use BASIC EVSE type instead of actual AeroViroment and ClipperCreek types.

    Attributes:
        See ChargingNetwork for Attributes.
    """
    def __init__(self, basic_evse=False):
        super().__init__()

        if basic_evse:
            evse_type = {'AV': 'BASIC', 'CC': 'BASIC'}
        else:
            evse_type = {'AV': 'AeroVironment', 'CC': 'ClipperCreek'}

        # Add Caltech EVSEs
        self.register_evse(get_evse_by_type('CA-148', evse_type['AV']))
        self.register_evse(get_evse_by_type('CA-148', evse_type['AV']))
        self.register_evse(get_evse_by_type('CA-149', evse_type['AV']))
        self.register_evse(get_evse_by_type('CA-212', evse_type['AV']))
        self.register_evse(get_evse_by_type('CA-213', evse_type['AV']))
        for i in range(303, 328):
            if i >= 320 or i <= 323:
                self.register_evse(get_evse_by_type('CA-' + str(i), evse_type['CC']))
            else:
                self.register_evse(get_evse_by_type('CA-' + str(i), evse_type['AV']))
        for i in range(489, 514):
            if i >= 493 or i <= 496:
                self.register_evse(get_evse_by_type('CA-' + str(i), evse_type['CC']))
            else:
                self.register_evse(get_evse_by_type('CA-' + str(i), evse_type['AV']))

        # Add Caltech Constraint Set
        CC_pod = Current(['CA-322', 'CA-493', 'CA-496', 'CA-320', 'CA-495', 'CA-321', 'CA-323', 'CA-494'])
        AV_pod = Current(['CA-324', 'CA-325', 'CA-326', 'CA-327', 'CA-489', 'CA-490', 'CA-491', 'CA-492'])
        AB = Current(['CA-{0}'.format(i) for i in [308, 508, 303, 513, 310, 506, 316, 500, 318, 498]]) + CC_pod + AV_pod
        BC = Current(['CA-{0}'.format(i) for i in [304, 512, 305, 511, 313, 503, 311, 505, 317, 499]])
        CA = Current(
            ['CA-{0}'.format(i) for i in [307, 509, 309, 507, 306, 510, 315, 501, 319, 497, 312, 504, 314, 502]])

        # Temporarily add ADA spaces to BC, since we don't know where they come from
        BC += Current(['CA-148', 'CA-149', 'CA-212', 'CA-213'])

        # Define the angles
        for load_id in AB.loads:
            self.register_load(load_id, math.radians(30))
        for load_id in BC.loads:
            self.register_load(load_id, math.radians(150))
        for load_id in CA.loads:
            self.register_load(load_id, math.radians(-90))

        # Define intermediate currents
        I3a = AB - CA
        I3b = BC - AB
        I3c = CA - BC
        I2a = (1 / 4) * (I3a - I3c)
        I2b = (1 / 4) * (I3b - I3a)
        I2c = (1 / 4) * (I3c - I3b)

        # Build constraint set
        self.add_constraint(CC_pod, 80)
        self.add_constraint(AV_pod, 80)
        self.add_constraint(I3a, 420)
        self.add_constraint(I3b, 420)
        self.add_constraint(I3c, 420)
        self.add_constraint(I2a, 180)
        self.add_constraint(I2b, 180)
        self.add_constraint(I2c, 180)
