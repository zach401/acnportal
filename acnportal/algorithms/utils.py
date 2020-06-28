import numpy as np


def infrastructure_constraints_feasible(
    rates,
    infrastructure,
    linear=False,
    violation_tolerance=1e-5,
    relative_tolerance=1e-7,
):
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
