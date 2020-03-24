# coding=utf-8
"""
Integration tests using gym_acnsim.
"""
import json
import os
import unittest
from importlib.util import find_spec
from typing import Dict, List

import numpy as np

from .test_integration import EarliestDeadlineFirstAlgoStateful, \
    TestIntegration
if find_spec("gym") is not None:
    from acnportal.acnsim import GymTrainedInterface
    from acnportal.acnsim.gym_acnsim.envs import make_default_sim_env
    from acnportal.algorithms import GymTrainedAlgorithm, SimRLModelWrapper


    class EDFWrapper(SimRLModelWrapper):
        """
        A model wrapper that mimics the behavior of the earliest deadline
        first (EDF) algorithm from the sorted_algorithms module. Instead of
        using the observation to generate the schedule, this algorithm
        instead directly uses the interface contained in the info parameter.
        """
        def __init__(self, model: object = None) -> None:
            super().__init__(model)
            self.edf_algo = EarliestDeadlineFirstAlgoStateful()
            self.polled_pilots = {}
            self.polled_charging_rates = {}

        def predict(self,
                    observation: Dict[str, np.ndarray],
                    reward: float,
                    done: bool,
                    info: Dict[str, GymTrainedInterface] = None) -> np.ndarray:
            """ Implements SimRLModelWrapper.predict() """
            # Generally, predictions in an OpenAI workflow do not use the
            # info returned after stepping the environment. We use the info
            # here to avoid having to rewrite the EDF algorithm in a way
            # that does not use the interface.
            if info is None:
                raise ValueError("EDFStepped model wrapper requires an "
                                 "info argument containing an Interface to "
                                 "run.")
            self.edf_algo.register_interface(info['interface'])
            schedule: Dict[str, List[float]] = self.edf_algo.run()
            if not info['interface'].current_time % 100:
                self.polled_pilots[str(info['interface'].current_time)] = \
                    info['interface'].last_applied_pilot_signals
                self.polled_charging_rates[
                    str(info['interface'].current_time)] = \
                    info['interface'].last_actual_charging_rate
            return np.array([
                schedule[station_id][0]
                if station_id in schedule
                else 0
                for station_id in self.edf_algo.interface.station_ids
            ]) - 16


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
class TestModelWrapperDeployed(TestIntegration):
    @classmethod
    def _build_scheduler(cls):
        cls.sch = GymTrainedAlgorithm()
        cls.sch.register_model(EDFWrapper())
        cls.env = make_default_sim_env()
        cls.sch.register_env(cls.env)
        return GymTrainedInterface

    def test_lap_interface_func(self):
        with open(os.path.join(os.path.dirname(__file__),
                               'edf_algo_pilot_signals.json')) as infile:
            self.edf_algo_true_lap = json.load(infile)

        if isinstance(self.sch.model, EDFWrapper):
            self.assertDictEqual(self.edf_algo_true_lap,
                                 self.sch.model.polled_pilots)

    def test_cr_interface_func(self):
        with open(os.path.join(os.path.dirname(__file__),
                               'edf_algo_charging_rates.json')) as infile:
            self.edf_algo_true_cr = json.load(infile)

        self.assertDictEqual(self.edf_algo_true_cr,
                             self.sch.model.polled_charging_rates)


if __name__ == '__main__':
    unittest.main()
