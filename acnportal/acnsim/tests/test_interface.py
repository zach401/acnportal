from typing import Any, Dict, List, Tuple
from unittest import TestCase
from unittest.mock import Mock, create_autospec, patch

import numpy as np

from .. import Simulator, Interface, InvalidScheduleError, EventQueue, \
    FiniteRatesEVSE, EV, EVSE, ChargingNetwork, GymTrainedInterface, \
    GymTrainingInterface


class TestInterface(TestCase):
    def setUp(self):
        self.simulator = create_autospec(Simulator)
        self.network = ChargingNetwork()
        self.simulator.network = self.network
        self.interface = Interface(self.simulator)
        self.evse1 = EVSE('PS-001')
        self.network.register_evse(self.evse1, 120, -30)
        self.evse2 = EVSE('PS-002')
        self.evse3 = EVSE('PS-003')
        self.network.register_evse(self.evse3, 360, 150)
        self.network.register_evse(self.evse2, 240, 90)

    def test_init(self):
        self.assertIs(self.interface._simulator, self.simulator)

    def test_active_evs(self):
        _ = self.interface.active_evs()
        self.simulator.get_active_evs.assert_called_once()

    def test_last_applied_pilot_signals_low_iteration(self):
        self.simulator.iteration = 1
        self.assertEqual(self.interface.last_applied_pilot_signals, {})

    def test_allowable_pilot_signals(self):
        self.assertEqual(
            self.interface.allowable_pilot_signals('PS-001'),
            (True, [0, float('inf')])
        )

    def test_evse_voltage(self):
        self.assertEqual(self.interface.evse_voltage('PS-002'), 240)

    def test_is_feasible_empty_schedule(self):
        self.assertTrue(self.interface.is_feasible({}))

    def test_is_feasible_unequal_schedules(self):
        with self.assertRaises(InvalidScheduleError):
            self.interface.is_feasible(
                {'PS-001' : [1, 2], 'PS-002' : [3, 4, 5], 'PS-003' : [4, 5]})

    def test_is_feasible(self):
        # Mock network's is_feasible function to check its call signature later
        self.network.is_feasible = create_autospec(self.network.is_feasible)
        self.interface.is_feasible({'PS-001' : [1, 2], 'PS-002' : [4, 5]})
        network_is_feasible_args = self.network.is_feasible.call_args
        # Check that the call to the network's is_feasible method has the correct arguments
        np.testing.assert_allclose(network_is_feasible_args[0][0], np.array([[1, 2], [0, 0], [4, 5]]))
        # Network's is_feasible method has its second argument (linear) defaulting to False. Check this is the case.
        self.assertEqual(network_is_feasible_args[0][1], False)
        # Network's is_feasible method has its third argument (violation_tolerance) defaulting to None. Check this is the case.
        self.assertIsNone(network_is_feasible_args[0][2])
        self.assertIsNone(network_is_feasible_args[0][3])

    def test_is_feasible_with_options(self):
        # Mock network's is_feasible function to check its call signature later
        self.network.is_feasible = create_autospec(self.network.is_feasible)
        self.interface.is_feasible({'PS-001' : [1, 2], 'PS-002' : [4, 5]}, linear=True, violation_tolerance=1e-3, relative_tolerance=1e-5)
        network_is_feasible_args = self.network.is_feasible.call_args
        # Check that the call to the network's is_feasible method has the correct arguments
        np.testing.assert_allclose(network_is_feasible_args[0][0], np.array([[1, 2], [0, 0], [4, 5]]))
        self.assertEqual(network_is_feasible_args[0][1], True)
        self.assertEqual(network_is_feasible_args[0][2], 1e-3)
        self.assertEqual(network_is_feasible_args[0][3], 1e-5)


class TestGymTrainedInterface(TestInterface):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        super().setUp()
        self.interface: GymTrainedInterface = \
            GymTrainedInterface.from_interface(self.interface)

    def test_station_ids(self) -> None:
        self.assertEqual(self.interface.station_ids,
                         ['PS-001', 'PS-003', 'PS-002'])

    def test_active_station_ids(self) -> None:
        # Auto-specs are of type Any as typing does not support Mocks.
        ev1: Any = create_autospec(EV)
        ev2: Any = create_autospec(EV)
        ev1.fully_charged = True
        ev2.fully_charged = False
        self.network.plugin(ev1, 'PS-001')
        self.network.plugin(ev2, 'PS-002')
        self.assertEqual(self.interface.active_station_ids,
                         ['PS-002'])

    def test_is_done(self) -> None:
        event_queue: EventQueue = EventQueue()
        event_queue.empty = Mock()
        self.simulator.event_queue = event_queue
        _ = self.interface.is_done
        event_queue.empty.assert_called_once()

    def test_charging_rates(self) -> None:
        self.simulator.charging_rates = np.eye(2)
        np.testing.assert_equal(self.interface.charging_rates, np.eye(2))

    def test_is_feasible_evse_key_error(self) -> None:
        with self.assertRaises(KeyError):
            self.interface.is_feasible_evse({'PS-001': [1], 'PS-000': [0]})

    def _continuous_evse_helper(self) -> None:
        self.evse1._max_rate = 32
        self.evse2._min_rate = 6
        self.evse3._min_rate = 6

    def test_is_feasible_evse_continuous_infeasible(self) -> None:
        self._continuous_evse_helper()
        schedule: Dict[str, List[float]] = {'PS-001': [34, 31],
                                            'PS-002': [4, 5],
                                            'PS-003': [0, 0]}
        self.assertFalse(self.interface.is_feasible_evse(schedule))

    def test_is_feasible_evse_continuous_feasible(self) -> None:
        self._continuous_evse_helper()
        schedule: Dict[str, List[float]] = {'PS-001': [31, 16],
                                            'PS-002': [7, 16],
                                            'PS-003': [0, 0]}
        self.assertTrue(self.interface.is_feasible_evse(schedule))

    def _discrete_evse_helper(self) -> None:
        self.evse1: FiniteRatesEVSE = FiniteRatesEVSE('PS-001',
                                                      [8, 16, 24, 32])
        self.evse2: FiniteRatesEVSE = FiniteRatesEVSE('PS-002',
                                                      [6, 16])
        self.evse3: FiniteRatesEVSE = FiniteRatesEVSE('PS-003',
                                                      list(range(1, 32)))
        self.network.register_evse(self.evse1, 120, -30)
        self.network.register_evse(self.evse3, 360, 150)
        self.network.register_evse(self.evse2, 240, 90)

    def test_is_feasible_evse_discrete_infeasible(self) -> None:
        self._discrete_evse_helper()
        schedule: Dict[str, List[float]] = {'PS-001': [4, 19],
                                            'PS-002': [8, 18],
                                            'PS-003': [0, 0]}
        self.assertFalse(self.interface.is_feasible_evse(schedule))

    def test_is_feasible_evse_discrete_feasible(self) -> None:
        schedule: Dict[str, List[float]] = {'PS-001': [8, 24],
                                            'PS-002': [6, 16],
                                            'PS-003': [0, 0]}
        self.assertTrue(self.interface.is_feasible_evse(schedule))

    @patch("acnportal.acnsim.Interface.is_feasible", return_value=True)
    def test_is_feasible(self, mocked_is_feasible) -> None:
        self.interface.is_feasible_evse = Mock()
        self.interface.is_feasible_evse.return_value = True
        self.assertTrue(self.interface.is_feasible({}))
        mocked_is_feasible.assert_called_once_with(
            {},
            linear=False,
            violation_tolerance=None,
            relative_tolerance=None
        )
        self.interface.is_feasible_evse.assert_called_once_with({})

    def test_last_energy_delivered(self) -> None:
        ev1: Any = create_autospec(EV)
        ev2: Any = create_autospec(EV)
        ev1.current_charging_rate = 32
        ev2.current_charging_rate = 16
        self.simulator.get_active_evs = Mock()
        self.simulator.get_active_evs.return_value = [ev1, ev2]
        self.assertEqual(self.interface.last_energy_delivered(), 48)

    @patch("acnportal.acnsim.ChargingNetwork.constraint_current",
           return_value=4-3j)
    def test_current_constraint_currents(self,
                                         mocked_constraint_current) -> None:
        self.assertEqual(self.interface.current_constraint_currents({}), 5)
        mocked_constraint_current.assert_called_once_with({}, time_indices=[0])


class TestGymTrainingInterface(TestGymTrainedInterface):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        super().setUp()
        self.interface: GymTrainingInterface = \
            GymTrainingInterface.from_interface(self.interface)
        self.simulator.step = Mock()
        self.simulator.step.return_value = True
        self.simulator.max_recompute = 2

    def test_step_warning_no_schedules(self) -> None:
        with self.assertWarns(UserWarning):
            self.interface.step({})

    def test_step_warning_short_schedule(self) -> None:
        self.simulator.max_recompute = 4
        with self.assertWarns(UserWarning):
            self.interface.step({'PS-001': [34, 31],
                                 'PS-002': [4, 5],
                                 'PS-003': [0, 0]})

    def _step_helper(self) -> Dict[str, List[float]]:
        schedule: Dict[str, List[float]] = {'PS-001': [34, 31],
                                            'PS-002': [4, 5],
                                            'PS-003': [0, 0]}
        event_queue: EventQueue = EventQueue()
        event_queue.empty = Mock()
        event_queue.empty.return_value = True
        self.simulator.event_queue = event_queue
        return schedule

    @patch("acnportal.acnsim.GymTrainingInterface.is_feasible",
           return_value=False)
    def test_step_infeasible_schedule(self, mocked_is_feasible) -> None:
        schedule: Dict[str, List[float]] = self._step_helper()
        self.assertEqual(self.interface.step(schedule), (True, False))
        self.simulator.event_queue.empty.assert_called_once()
        mocked_is_feasible.assert_called_once_with(schedule)
        self.simulator.step.assert_not_called()

    @patch("acnportal.acnsim.GymTrainingInterface.is_feasible",
           return_value=True)
    def test_step_feasible_schedule(self, mocked_is_feasible) -> None:
        schedule: Dict[str, List[float]] = self._step_helper()
        self.assertEqual(self.interface.step(schedule), (True, True))
        self.simulator.event_queue.empty.assert_not_called()
        mocked_is_feasible.assert_called_once_with(schedule)
        self.simulator.step.assert_called_once_with(schedule)

    @patch("acnportal.acnsim.GymTrainingInterface.is_feasible",
           return_value=False)
    def test_step_infeasible_schedule_no_force_feasibility(
            self, mocked_is_feasible) -> None:
        schedule: Dict[str, List[float]] = self._step_helper()
        self.assertEqual(self.interface.step(schedule,
                                             force_feasibility=False),
                         (True, False))
        self.simulator.event_queue.empty.assert_not_called()
        mocked_is_feasible.assert_called_once_with(schedule)
        self.simulator.step.assert_called_once_with(schedule)
