# coding=utf-8
""" Functions to generate Sessions and Infrastructures for testing algorithms. """
from typing import List, Iterable, Optional, Dict, Union

import numpy as np

SessionDict = Dict[str, Optional[Union[str, float, int, Iterable[float]]]]


def session_generator(
    num_sessions: int,
    arrivals: List[int],
    departures: List[int],
    requested_energy: List[float],
    remaining_energy: List[float],
    max_rates: List[Union[float, Iterable[float], np.ndarray]],
    min_rates: Optional[List[Union[float, Iterable[float], np.ndarray]]] = None,
    station_ids: Optional[List[str]] = None,
    estimated_departures: Optional[List[Union[float, Iterable[float], np.ndarray]]] = None
) -> List[SessionDict]:
    """ Generate Sessions with the input info in dict format. """
    sessions: List[SessionDict] = []
    min_rates: List[float] = min_rates if min_rates is not None else [0] * num_sessions
    for i in range(num_sessions):
        station_id: str = station_ids[i] if station_ids is not None else f"{i}"
        session_id: str = f"{i}"
        s: SessionDict = {
            "station_id": station_id,
            "session_id": session_id,
            "requested_energy": requested_energy[i],
            "energy_delivered": requested_energy[i] - remaining_energy[i],
            "arrival": arrivals[i],
            "departure": departures[i],
            "estimated_departure": estimated_departures[i] if estimated_departures is not None else departures[i],
            "min_rates": min_rates[i],
            "max_rates": max_rates[i],
        }
        sessions.append(s)
    return sessions


InfrastructureDict = Dict[str, Union[np.ndarray, List[str], List[np.ndarray]]]


def single_phase_single_constraint(
    num_evses: int,
    limit: float,
    max_pilot: float = 32,
    min_pilot: float = 8,
    allowable_pilots: Optional[List[np.ndarray]] = None,
    is_continuous: Optional[np.ndarray] = None,
) -> InfrastructureDict:
    """ Generates a single-phase one-constraint network; returns the corresponding
    infrastructure info in a dict. """
    if allowable_pilots is None:
        allowable_pilots: List[np.ndarray] = [
            np.array([min_pilot, max_pilot])
        ] * num_evses
    if is_continuous is None:
        is_continuous: np.ndarray = np.ones(num_evses, dtype=bool)
    infrastructure: InfrastructureDict = {
        "constraint_matrix": np.ones((1, num_evses)),
        "constraint_limits": np.array([limit]),
        "phases": np.zeros(num_evses),
        "voltages": np.repeat(208, num_evses),
        "constraint_ids": ["all"],
        "station_ids": [f"{i}" for i in range(num_evses)],
        "max_pilot": np.repeat(max_pilot, num_evses),
        "min_pilot": np.repeat(min_pilot, num_evses),
        "allowable_pilots": allowable_pilots,
        "is_continuous": is_continuous,
    }
    return infrastructure


def three_phase_balanced_network(
    evses_per_phase: int,
    limit: float,
    max_pilot: float = 32,
    min_pilot: float = 8,
    allowable_pilots: Optional[List[Iterable[float]]] = None,
    is_continuous: Optional[np.ndarray] = None,
) -> InfrastructureDict:
    """ Generates a 3-phase 3-constraint network; returns the corresponding
    infrastructure info in a dict. """
    n: int = evses_per_phase
    num_evses: int = 3 * evses_per_phase
    if allowable_pilots is None:
        allowable_pilots: List[np.ndarray] = [
            np.array([min_pilot, max_pilot])
        ] * num_evses
    if is_continuous is None:
        is_continuous: np.ndarray = np.ones(num_evses, dtype=bool)
    infrastructure: InfrastructureDict = {
        "constraint_matrix": np.array(
            [
                [1] * n + [-1] * n + [0] * n,
                [0] * n + [1] * n + [-1] * n,
                [-1] * n + [0] * n + [1] * n,
            ]
        ),
        "constraint_limits": np.repeat(limit, 3),
        "phases": np.array([0] * n + [-120] * n + [120] * n),
        "voltages": np.repeat(208, 3 * evses_per_phase),
        "constraint_ids": ["AB", "BC", "CA"],
        "station_ids": [f"{i}" for i in range(num_evses)],
        "max_pilot": np.repeat(max_pilot, num_evses),
        "min_pilot": np.repeat(min_pilot, num_evses),
        "allowable_pilots": allowable_pilots,
        "is_continuous": is_continuous,
    }
    return infrastructure
