import numpy as np

from acnportal.acnsim.interface import InfrastructureInfo, SessionInfo


def infrastructure_constraints_feasible(
    rates: np.ndarray,
    infrastructure: InfrastructureInfo,
    linear: bool = False,
    violation_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-7,
) -> bool:
    tol = np.maximum(
        violation_tolerance, relative_tolerance * infrastructure.constraint_limits
    )
    if not linear:
        phase_in_rad = np.deg2rad(infrastructure.phases)
        for j, v in enumerate(infrastructure.constraint_matrix):
            a = np.stack([v * np.cos(phase_in_rad), v * np.sin(phase_in_rad)])
            line_currents = np.linalg.norm(a @ rates, axis=0)
            if not np.all(
                line_currents <= infrastructure.constraint_limits[j] + tol[j]
            ):
                return False
    else:
        for j, v in enumerate(infrastructure.constraint_matrix):
            line_currents = np.linalg.norm(np.abs(v) @ rates, axis=0)
            if not np.all(
                line_currents <= infrastructure.constraint_limits[j] + tol[j]
            ):
                return False
    return True


def remaining_amp_periods(
    session: SessionInfo, infrastructure: InfrastructureInfo, period: float
) -> float:
    i = infrastructure.get_station_index(session.station_id)
    amp_hours = session.remaining_demand * 1000 / infrastructure.voltages[i]
    return amp_hours * 60 / period
