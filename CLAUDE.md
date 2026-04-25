# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install (development):**
```bash
pip install .[all]
```

**Run all tests:**
```bash
pytest
```

**Run a single test file:**
```bash
pytest acnportal/acnsim/models/tests/test_EV.py
```

**Run a specific test case or method:**
```bash
pytest acnportal/acnsim/models/tests/test_EV.py::TestEV::test_arrival
```

**Lint / format:**
```bash
flake8 acnportal/
black acnportal/
```

CI runs `pytest` against Python 3.8–3.11. PRs should target the `dev` branch, not `master`.

## Architecture Overview

ACN Portal is a Python research toolkit for EV charging. The main packages live under `acnportal/`:

### `acnsim` — Simulation engine

The central object is `Simulator` (`acnportal/acnsim/simulator.py`). It owns a `ChargingNetwork`, a `BaseAlgorithm`, and an `EventQueue`, and drives the simulation loop:

1. Pop current `Event`s from the queue and process them (Plugin/Unplug/Recompute).
2. If a schedule recompute is needed, call `scheduler.run()` → `schedule()`.
3. Push pilot signals to the network via `network.update_pilots()`.
4. Store actual charging rates back from the network.

**Algorithm ↔ Simulator decoupling via `Interface`** (`acnportal/acnsim/interface.py`): Algorithms never touch the `Simulator` directly. Instead, they interact only through an `Interface` object registered during setup (`scheduler.register_interface(Interface(sim))`). The interface exposes `active_sessions()` (returning `SessionInfo` objects), `infrastructure_info()`, `is_feasible()`, and price/demand-charge signals. This means the same algorithm class can run on simulated or real hardware without changes.

**Model layer** (`acnportal/acnsim/models/`):
- `EV` — tracks arrival, departure, requested/delivered energy, and delegates to a `Battery` for SoC updates.
- `BaseEVSE` / subclasses (`acnportal/acnsim/models/evse.py`) — represent individual charging stations with `min_rate`/`max_rate` and discrete-or-continuous pilot signal support.
- `Battery` hierarchy (`acnportal/acnsim/models/battery.py`) — models charge acceptance.

**Network** (`acnportal/acnsim/network/charging_network.py`): `ChargingNetwork` holds an `OrderedDict` of EVSEs and a linear constraint matrix relating individual station currents to aggregate currents (per phase, per transformer, etc.). `is_feasible()` checks whether a proposed schedule violates any network constraint. Pre-built real-world site topologies live in `acnportal/acnsim/network/sites/`.

**Events** (`acnportal/acnsim/events/`): `PluginEvent`, `UnplugEvent`, `RecomputeEvent` extend `Event`. Events are stored in a priority-queue `EventQueue`. `acndata_events.py` builds event queues from real ACN-Data session records; `stochastic_events.py` generates synthetic arrivals.

**Serialization** (`acnportal/acnsim/base.py`): All simulation objects inherit from `BaseSimObj`, which provides `to_json()` / `from_json()` round-tripping via a `_to_dict` / `_from_dict` protocol. Objects are stored in a shared `context_dict` keyed by Python `id()` so that objects referenced from multiple places are not duplicated on load.

**Analysis** (`acnportal/acnsim/analysis/`): Post-run helper functions that operate on a completed `Simulator` — aggregate current/power, energy delivered, constraint currents, cost under a tariff, etc.

### `algorithms` — Scheduling algorithms

All algorithms subclass `BaseAlgorithm` (`acnportal/algorithms/base_algorithm.py`) and implement a single method:

```python
def schedule(self, active_sessions: List[SessionInfo]) -> Dict[str, List[float]]:
    ...
```

The return value maps `station_id → list of pilot signals`, one entry per future period. Set `self.max_recompute = 1` to force re-scheduling every period.

`SortedSchedulingAlgo` (`acnportal/algorithms/sorted_algorithms.py`) implements FCFS, EDF, and similar algorithms by sorting sessions with a pluggable `sort_fn`, then binary-searching for the maximum feasible rate per EV. Helper modules `preprocessing.py` and `postprocessing.py` provide composable functions (enforce pilot limits, apply upper-bound estimates, round to allowable discrete rates, etc.).

### `acndata` — Data client

`DataClient` (`acnportal/acndata/data_client.py`) is a thin wrapper around the ACN-Data REST API. Pass an API token and call `get_sessions(site, ...)` to iterate over historical charging sessions from `caltech`, `jpl`, or `office001`.

### `signals` — External signals

`signals/tariffs/` contains `TimeOfUseTariff` / `TariffSchedule` classes and bundled JSON tariff schedules (PG&E A-10, SCE TOU-EV-4/8). The `Simulator` accepts a `signals` dict; the `Interface` exposes `get_prices()` and `get_demand_charge()` when a `"tariff"` key is present.

### `contrib` — Community extensions

`contrib/acnsim/network/stochastic_network.py` provides a `StochasticNetwork` that overrides `plugin()` to assign arriving EVs to a randomly chosen available EVSE rather than a fixed station. EVs that arrive when all stations are occupied are held in a waiting queue and assigned when a spot opens up.

## Key Conventions

- **Docstrings**: Google-style. Place class docstrings on the class declaration, not in `__init__`.
- **Imports**: Use relative imports within the package (`from .models import EV`, not `from acnportal.acnsim.models import EV`).
- **Private attributes**: Prefix with `_`. Cross-module access of private attributes is allowed but must be suppressed with a `# noinspection PyProtectedMember` annotation and a comment explaining why.
- **Type hints**: Required on all new code.
- **f-strings**: Prefer over `.format()`.
- **LSP compliance**: Test subclasses should inherit from a shared `TestCase` subclass rather than directly from `unittest.TestCase` to ensure substitutability.
- **Backwards compatibility**: Deprecate removed arguments or renamed attributes rather than deleting them immediately; use `warnings.warn(..., DeprecationWarning)`.
- **Extending serialization**: If you add a new `BaseSimObj` subclass, implement both `_to_dict` and `_from_dict` so the JSON round-trip remains lossless.
