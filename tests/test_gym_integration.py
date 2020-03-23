# coding=utf-8
"""
Integration tests using gym_acnsim.
"""
import json
import os
import unittest
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pytz

from acnportal.acnsim import Simulator, GymTrainedInterface
from acnportal.acnsim.events import acndata_events
from acnportal.acnsim.network import sites
from acnportal.algorithms import GymTrainedAlgorithm, SimRLModelWrapper, \
    SortedSchedulingAlgo, earliest_deadline_first, BaseAlgorithm
from .test_integration import EarliestDeadlineFirstAlgoStateful, \
    TestIntegration


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
        return np.array([
            schedule[station_id][0]
            if station_id in schedule
            else 0
            for station_id in self.edf_algo.interface.station_ids
        ])


# class TestModelWrapperDeployed(TestIntegration):
#     @classmethod
#     def _build_scheduler(cls):
#         cls.sch = GymTrainedAlgorithm()
#         cls.sch.register_model(EDFWrapper())
#         cls.sch.register_env()


if __name__ == '__main__':
    unittest.main()
