# coding=utf-8
"""
Classes for generating Events from stochastic models.
"""
from typing import List, Dict, Any
from acnportal.acnsim.events import EventQueue, PluginEvent
from acnportal.acnsim.models import EV, Battery
import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd


class StochasticEvents:
    """ Base class for generating events from a stochastic model.
    
    Args:
        arrival_min (float): Lower bound for arrival time, [hours since midnight]
        arrival_max (float): Upper bound for arrival time [hours since midnight]
        duration_min (float): Lower bound for duration [hours] 
        duration_max (float): Upper bound for duration [hours]  
        energy_min (float): Lower bound for energy request [kWh] 
        energy_max (float): Upper bound for energy request [kWh]     
    """

    arrival_min: float
    arrival_max: float
    duration_min: float
    duration_max: float
    energy_min: float
    energy_max: float

    def __init__(
        self,
        arrival_min: float = 0.0,
        arrival_max: float = 24.0,
        duration_min: float = 0.0833,
        duration_max: float = 48.0,
        energy_min: float = 0.5,
        energy_max: float = 150.0,
    ) -> None:
        self.arrival_min = arrival_min
        self.arrival_max = arrival_max
        self.duration_min = duration_min
        self.duration_max = duration_max
        self.energy_min = energy_min
        self.energy_max = energy_max

    def fit(self, data: List[Dict[str, Any]]) -> None:
        """ Fit StochasticEvents model to data from ACN-Data.

        Args:
            data (List[Dict[str, Any]]): List of session dictionaries.
                See DataClient.get_sessions().

        Returns:
            None
        """
        pass

    def sample(self, n_samples: int) -> np.ndarray:
        """ Generate random samples from the fitted model.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            np.ndarray: shape (n_samples, 3), randomly generated samples. Column 1 is
                the arrival time in hours since midnight, column 2 is the sojourn
                time in hours, and column 3 is the energy demand in kWh.
        """
        pass

    def clip_samples(self, sample_matrix: np.ndarray) -> np.ndarray:
        """ Clip samples matrix into their upper and lower bounds.

            Note that this function will modify the samples_matrix in place as well as
            return it.

        Args:
            sample_matrix (np.ndarray): shape (n_samples, 3), randomly generated
                samples. Column 1 is the arrival time in hours since midnight,
                column 2 is the sojourn time in hours, and column 3 is the energy
                demand in kWh.

        Returns:
            np.ndarray: sample matrix with all entries projected into their
                corresponding bounds.
        """
        sample_matrix[:, 0] = np.clip(
            sample_matrix[:, 0], self.arrival_min, self.arrival_max
        )
        sample_matrix[:, 1] = np.clip(
            sample_matrix[:, 1], self.duration_min, self.duration_max
        )
        sample_matrix[:, 2] = np.clip(
            sample_matrix[:, 2], self.energy_min, self.energy_max
        )
        return sample_matrix

    def generate_events(
        self,
        sessions_per_day: List[int],
        period: float,
        voltage: float,
        max_battery_power: float,
        max_len: int = None,
        battery_params: Dict[str, Any] = None,
        force_feasible: bool = False,
    ) -> EventQueue:
        """ Return EventQueue from random generated samples.

            Args:
                sessions_per_day (List[int]): Number of sessions to sample for each day
                    of the simulation.
                period (int): Length of each time interval. (minutes)
                voltage (float): Voltage of the network.
                max_battery_power (float): Default maximum charging power for batteries.
                max_len (int): Maximum length of a session. (periods) Default None.
                battery_params (Dict[str, object]): Dictionary containing parameters for
                    the EV's battery. Three keys are supported. If none, Battery type
                    is used with default configuration. Default None.
                        - 'type' maps to a Battery-like class. (required)
                        - 'capacity_fn' maps to a function which takes in the the energy
                            delivered to the car, the length of the session,
                            the period of the simulation, and the voltage of the
                            system. It should return a tuple with the capacity of the
                            battery and the initial charge of the battery both in
                            A*periods.
                        - 'kwargs' maps to a dictionary of keyword arguments which will
                            be based to the Battery constructor.
                force_feasible (bool): If True, the requested_energy of each session
                    will be reduced if it exceeds the amount of energy which could be
                    delivered at maximum rate during the duration of the charging
                    session. Default False. Returns: EventQueue: Queue of plugin
                    events for the samples charging sessions.
        """
        daily_sessions = []
        for d, num_sessions in enumerate(sessions_per_day):
            if sessions_per_day[d] > 0:
                daily_arrivals = self.sample(num_sessions)
                daily_arrivals[:, 0] += 24 * d
                daily_sessions.append(daily_arrivals)
        ev_matrix = np.vstack([day for day in daily_sessions if day is not None])
        evs = self._convert_ev_matrix(
            ev_matrix,
            period,
            voltage,
            max_battery_power,
            max_len,
            battery_params,
            force_feasible,
        )
        events = [PluginEvent(sess.arrival, sess) for sess in evs]
        return EventQueue(events)

    @staticmethod
    def extract_training_data(data: List[Dict[str, Any]]):
        """ Generate matrix for training Gaussian Mixture Model.

        Args:
            data (List[Dict[str, Any]]): List of session dictionaries.
                See DataClient.get_sessions().

        Returns:
            np.ndarray: shape(n_sessions, 3) Column 1 is the arrival time in hours since
                midnight, column 2 is the sojourn time in hours, and column 3 is the
                energy demand in kWh.
        """
        df = pd.DataFrame(data)
        df.sort_values(by="connectionTime", inplace=True)
        connection_time = [v.hour + v.minute / 60 for v in df["connectionTime"]]
        durations = [
            v.total_seconds() / 3600
            for v in df["disconnectTime"] - df["connectionTime"]
        ]
        energy = [v for v in df["kWhDelivered"]]
        return np.array([connection_time, durations, energy]).T

    @staticmethod
    def _convert_ev_matrix(
        ev_matrix: np.ndarray,
        period: float,
        voltage: float,
        max_battery_power: float,
        max_len: int = None,
        battery_params: Dict[str, Any] = None,
        force_feasible: bool = False,
    ) -> List[EV]:
        """

        Args:
            ev_matrix (np.ndarray[float]): Nx3 array where N is the number of EVs.
                Column 1 is the arrival time in hours since midnight, column 2 is the
                sojourn time in hours, and column 3 is the energy demand in kWh.
            (See generate_events() for other arguments)

        Returns:
            List[EV]: List of EVs with parameters taken from ev_matrix.
        """

        period_per_hour = 60 / period
        evs = []
        for row_idx, row in enumerate(ev_matrix):
            arrival, sojourn, energy_delivered = row

            if arrival < 0 or sojourn <= 0 or energy_delivered <= 0:
                print("Invalid session.")
                continue

            if max_len is not None and sojourn > max_len:
                sojourn = max_len

            if force_feasible:
                max_feasible = max_battery_power * sojourn
                energy_delivered = np.minimum(max_feasible, energy_delivered)

            departure = int((arrival + sojourn) * period_per_hour)
            arrival = int(arrival * period_per_hour)
            session_id = "session_{0}".format(row_idx)
            # By default a new station is created for each EV.
            # Infinite space assumption.
            station_id = "station_{0}".format(row_idx)

            if battery_params is None:
                battery_params = {"type": Battery}
            batt_kwargs = battery_params["kwargs"] if "kwargs" in battery_params else {}
            if "capacity_fn" in battery_params:
                cap_fn = battery_params["capacity_fn"]
                cap, init = cap_fn(energy_delivered, sojourn, voltage, period)
            else:
                cap = energy_delivered
                init = 0
            battery_type = battery_params["type"]
            battery = battery_type(cap, init, max_battery_power, **batt_kwargs)
            evs.append(
                EV(
                    arrival,
                    departure,
                    energy_delivered,
                    station_id,
                    session_id,
                    battery,
                )
            )
        return evs


class GaussianMixtureEvents(StochasticEvents):
    """ Model to draw charging session parameters from a gaussian mixture model.

    Args:
        pretrained_model (GaussianMixture): A trained Gaussian Mixture Model with
            variables arrival time (h), sojourn time (h), energy demand (kWh).
        Also accepts any kwargs for the sklearn GaussianMixture class.
        See https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html.
    """

    def __init__(
        self,
        arrival_min: float = 0.0,
        arrival_max: float = 24.0,
        duration_min: float = 0.0833,
        duration_max: float = 48.0,
        energy_min: float = 0.5,
        energy_max: float = 150.0,
        pretrained_model=None,
        **kwargs
    ):
        super().__init__(
            arrival_min, arrival_max, duration_min, duration_max, energy_min, energy_max
        )
        if pretrained_model is None:
            self.gmm = GaussianMixture(**kwargs)
        else:
            self.gmm = pretrained_model

    def fit(self, data: List[Dict[str, Any]], **kwargs) -> None:
        """ Fit StochasticEvents model to data from ACN-Data.

        Args:
            data (List[Dict[str, Any]]): List of session dictionaries.
            See DataClient.get_sessions().

        Returns:
            None
        """
        x = self.extract_training_data(data)
        self.gmm.fit(x, **kwargs)

    def sample(self, n_samples: int):
        """ Generate random samples from the fitted model.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            np.ndarray: shape (n_samples, 3), randomly generated samples. Column 1 is
                the arrival time in hours since midnight, column 2 is the sojourn
                time in hours, and column 3 is the energy demand in kWh.
        """
        if n_samples > 0:
            ev_matrix, _ = self.gmm.sample(n_samples)
            return self.clip_samples(ev_matrix)
        else:
            return np.array([])
