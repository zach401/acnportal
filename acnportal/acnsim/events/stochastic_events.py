from ..models.ev import EV
from ..models.battery import Battery
from . import PluginEvent
from .event_queue import EventQueue
import numpy as np


class StochasticSessionModel:
    def get_sessions(self, num_sessions):
        raise NotImplementedError("StochasticEventModel is an abstract class.")


class GaussianMixtureSessionModel(StochasticSessionModel):
    """ Model to draw charging session parameters from a gaussian mixture model.

    Args:
        gmm (scikitlearn.GaussianMixture): A trained Gaussian Mixure Model with
            variables arrival time (h), sojourn time (h), energy demand (kWh).
    """
    def __init__(self, gmm):
        self.gmm = gmm

    def get_sessions(self, num_sessions):
        if num_sessions > 0:
            daily_arrivals, _ = self.gmm.sample(num_sessions)
            return daily_arrivals
        else:
            return None


def stochastic_events(model, arrivals_per_day, period, voltage, max_battery_power,
                      **kwargs):
    """ Return EventQueue filled using events gathered from the acndata API.

    Args:
        model (StochasticSessionModel): Model from which to draw session data.
        arrivals_per_day (List[int]): Number of arrivals for each day.
        period (int): Length of each time interval. (minutes)
        voltage (float): Voltage of the network.
        max_battery_power (float): Default maximum charging power for batteries.
        max_len (int): Maximum length of a session. (periods) Default None.
        battery_params (Dict[str, object]): Dictionary containing parameters for the EV's battery. Three keys are
            supported. If none, Battery type is used with default configuration. Default None.
            - 'type' maps to a Battery-like class. (required)
            - 'capacity_fn' maps to a function which takes in the the energy delivered to the car, the length of the
                session, the period of the simulation, and the voltage of the system. It should return a tuple with
                the capacity of the battery and the initial charge of the battery both in A*periods.
            - 'kwargs' maps to a dictionary of keyword arguments which will be based to the Battery constructor.
        force_feasible (bool): If True, the requested_energy of each session will be reduced if it exceeds the amount
            of energy which could be delivered at maximum rate during the duration of the charging session.
            Default False.

    Returns:
        EventQueue: An EventQueue filled with Events gathered through the acndata API.
    """
    daily_sessions = []
    for d in range(len(arrivals_per_day)):
        if arrivals_per_day[d] > 0:
            daily_arrivals = model.get_sessions(arrivals_per_day[d])
            daily_arrivals[:, 0] += 24 * d
            daily_sessions.append(daily_arrivals)
    ev_matrix = np.vstack([day for day in daily_sessions if day is not None])
    evs = _convert_ev_matrix(ev_matrix, period, voltage, max_battery_power, **kwargs)
    events = [PluginEvent(sess.arrival, sess) for sess in evs]
    return EventQueue(events)


def assign_random_stations(events):
    """ Assign random stations to each plugin event.

    Args:
        events (EventQueue): An event queue.

    Returns:
        EventQueue: An EventQueue with random spaces assigned.

    """
    T = max(
        event.ev.departure
        for timestamp, event in events._queue
        if isinstance(events, PluginEvent)
    )
    assignment_matrix = np.ones(shape=(1, T))
    evs = []
    for arrival, event in events._queue:
        if isinstance(events, PluginEvent):
            ev = event.ev
            available = np.all(assignment_matrix[:, ev.arrival : ev.departure], axis=1)
            if not np.any(available):
                assignment_matrix = np.vstack(assignment_matrix, np.ones(shape=(1, T)))
                assignment = assignment_matrix.shape[1]
            else:
                assignment = np.random.choice(np.where(available)[0])
            ev.station_id = "station_{0}".format(assignment)
            evs.append(ev)
            assignment_matrix[assignment, ev.arrival : ev.departure] = 0
    return EventQueue([PluginEvent(ev.arrival, ev) for ev in evs])


def _convert_ev_matrix(
    ev_matrix,
    period,
    voltage,
    max_battery_power,
    max_len=None,
    battery_params=None,
    force_feasible=False,
):
    """

    Args:
        ev_matrix (np.ndarray[float]): Nx3 array where N is the number of EVs. Column 1 is the arrival time in hours
            since midnight, column 2 is the sojourn time in hours, and column 3 is the energy demand in kWh.
        period (int): Length of each time interval. (minutes)
        voltage (float): Voltage of the network.
        max_battery_power (float): Default maximum charging power for batteries.
        max_len (int): Maximum length of a session. (periods) Default None.
        battery_params (Dict[str, object]): Dictionary containing parameters for the EV's battery. Three keys are
            supported. If none, Battery type is used with default configuration. Default None.
            - 'type' maps to a Battery-like class. (required)
            - 'capacity_fn' maps to a function which takes in the the energy delivered to the car, the length of the
                session, the period of the simulation, and the voltage of the system. It should return a tuple with
                the capacity of the battery and the initial charge of the battery both in A*periods.
            - 'kwargs' maps to a dictionary of keyword arguments which will be based to the Battery constructor.
        force_feasible (bool): If True, the requested_energy of each session will be reduced if it exceeds the amount
            of energy which could be delivered at maximum rate during the duration of the charging session.
            Default False.

    Returns:

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
        # By default a new station is created for each EV. Infinite space assumption.
        station_id = "station_{0}".format(row_idx)

        if battery_params is None:
            battery_params = {"type": Battery}
        batt_kwargs = battery_params["kwargs"] if "kwargs" in battery_params else {}
        if "capacity_fn" in battery_params:
            cap, init = battery_params["capacity_fn"](
                energy_delivered, sojourn, voltage, period
            )
        else:
            cap = energy_delivered
            init = 0
        batt = battery_params["type"](cap, init, max_battery_power, **batt_kwargs)
        evs.append(EV(arrival, departure, energy_delivered, station_id, session_id, batt))
    return evs
