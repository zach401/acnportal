from algorithms.base_algorithm import BaseAlgorithm
from contrib.hardcoded_constraints.gurobi_utils import OptimizationScheduler, OptimizationPostProcessor, InfeasibilityException, PHASE_AWARE
import math
import numpy as np
from collections import defaultdict
from copy import deepcopy


FORCE_UNIQUE_COEFF = 1e-7


def schedule_dict_to_list(sch, T):
    sch_list = [sch[t] if t in sch else 0 for t in range(T)]
    return sch_list


class OptScheduler(BaseAlgorithm):
    def __init__(self, const_scale=1, const_type=PHASE_AWARE, init_peak=0, force_unique=False, energy_equality=False):
        super().__init__()
        self.const_scale = const_scale
        self.const_type = const_type
        self.init_peak = init_peak
        self.force_unique = force_unique
        self.energy_equality = energy_equality

    def obj(self, scheduler, length):
        raise NotImplementedError('obj() is not defined in the base class. Please use a derived class.')

    def _solve(self, active_evs):
        if len(active_evs) == 0:
            return {}

        t = self.interface.get_current_time()
        for ev in active_evs:
            ev.arrival -= t
            ev.departure -= t
        T = int(math.ceil(np.max([ev.departure for ev in active_evs]))) + 1

        sch = OptimizationScheduler(const_type=PHASE_AWARE, const_scale=self.const_scale, energy_equality=self.energy_equality)
        sch.add_evs(active_evs)
        sch.obj = self.obj(sch, T)

        sch.compile()
        obj, schedule, runtime = sch.solve()
        return schedule

    def schedule(self, active_evs):
        if len(active_evs) == 0:
            return {}

        try:
            schedule = self._solve(active_evs)
        except InfeasibilityException:
            print('An error occured at time {0}'.format(self.interface.get_current_time()))
            return {}

        T = int(math.ceil(np.max([ev.departure for ev in active_evs]))) + 1
        ev_to_evse = {ev.session_id: ev.station_id for ev in active_evs}
        full_schedule = {ev_to_evse[key]: np.clip(schedule_dict_to_list(val, T), a_min=0, a_max=None) for key, val in schedule.items()}
        return full_schedule


class ProfitMaximization(OptScheduler):
    def __init__(self, dc_proportional=False, const_scale=1, const_type=PHASE_AWARE, init_peak=0, force_unique=False, energy_equality=False):
        super().__init__(const_scale, const_type, init_peak, force_unique, energy_equality)
        self.dc_proportional = dc_proportional

    def get_peak(self):
        return max(self.interface.get_prev_peak(), self.init_peak)

    def obj(self, scheduler, length):
        t = self.interface.get_current_time()
        revenue = self.interface.get_revenue()
        prices = self.interface.get_prices(t, length)
        if self.dc_proportional:
            # demand_charge = self.interface.get_demand_charge(length)
            demand_charge = self.interface.simulator.prices.demand_charge / 30
        else:
            demand_charge = self.interface.simulator.prices.demand_charge
        prev_peak = self.get_peak()
        temp_obj = scheduler.max_profit_obj(revenue, prices, demand_charge, prev_peak)
        if self.force_unique:
            temp_obj += FORCE_UNIQUE_COEFF*scheduler.force_concave()
        return temp_obj


class ProfitMaximizationWithQuickCharge(ProfitMaximization):
    def __init__(self, alpha, dc_proportional=False, const_scale=1, const_type=PHASE_AWARE, init_peak=0, force_unique=False, energy_equality=False):
        super().__init__(dc_proportional, const_scale, const_type, init_peak, force_unique, energy_equality)
        self.alpha = alpha

    def obj(self, scheduler, length):
        temp_obj = super().obj(scheduler, length)
        temp_obj += (self.alpha/length)*scheduler.quick_charge_obj()
        return temp_obj


class ProfitMaximizationSolar(ProfitMaximizationWithQuickCharge):
    def __init__(self, alpha, solar, dc_proportional=False, const_scale=1, const_type=PHASE_AWARE, init_peak=0, force_unique=False, energy_equality=False):
        super().__init__(alpha, dc_proportional, const_scale, const_type, init_peak, force_unique, energy_equality)
        self.solar_signal = solar

    def obj(self, scheduler, length):
        t = self.interface.get_current_time()
        revenue = self.interface.get_revenue()
        prices = self.interface.get_prices(t, length)
        demand_charge = self.interface.get_demand_charge(length)
        prev_peak = self.get_peak()
        solar_gen = self.solar_signal.get_generation(t, length)
        temp_obj = scheduler.max_profit_w_solar_obj(revenue, prices, demand_charge, solar_gen, prev_peak)
        if self.force_unique:
            temp_obj += FORCE_UNIQUE_COEFF*scheduler.force_concave()
        if self.alpha > 0:
            temp_obj += (self.alpha/length)*scheduler.quick_charge_obj()
        return temp_obj


class GreedySchedule(OptScheduler):
    def obj(self, scheduler, length):
        temp_obj = scheduler.quick_charge_obj()
        if self.force_unique:
            temp_obj += FORCE_UNIQUE_COEFF*scheduler.force_concave()
        return temp_obj


class TrackSignal(OptScheduler):
    def __init__(self, solar, alpha=0, const_scale=1, const_type=PHASE_AWARE, init_peak=0, force_unique=False, energy_equality=False):
        super().__init__(const_scale, const_type, init_peak, force_unique, energy_equality)
        self.solar_signal = solar
        self.alpha = alpha

    def obj(self, scheduler, length):
        t = self.interface.get_current_time()
        signal = self.solar_signal.get_generation(t, length)
        temp_obj = scheduler.tracking_obj(signal, self.alpha)
        # if self.alpha > 0:
        #     temp_obj += self.alpha*scheduler.quick_charge_obj()
        # if self.force_unique:
        #     temp_obj += FORCE_UNIQUE_COEFF * scheduler.force_concave()
        return temp_obj


def project_rate(rate, min_rate=6, eps=0.1):
    proj_rate = int(rate + eps)
    if proj_rate < min_rate:
        proj_rate = 0
    return proj_rate


class PostProcessor:
    def __init__(self, min_rate=6, cutoff=6, const_scale=1, const_type=PHASE_AWARE, init_peak=0):
        super().__init__()
        self.const_scale = const_scale
        self.const_type = const_type
        self.init_peak = init_peak
        self.min_rate = min_rate
        self.cutoff = cutoff

    def post_process(self, active_evs, schedule):
        if len(active_evs) == 0:
            return {}

        evs = {ev.session_id: deepcopy(ev) for ev in active_evs}

        schedule_by_time = defaultdict(dict)
        for id in schedule:
            for t in schedule[id]:
                schedule_by_time[t][id] = schedule[id][t]

        T_min = min(schedule_by_time.keys())
        T_max = max(schedule_by_time.keys())

        processed_schedule = defaultdict(dict)
        e_del = {ev.session_id: 0 for ev in active_evs}
        for t in range(T_min, T_max+1):
            if t not in schedule_by_time or len(schedule_by_time[t]) == 0:
                continue
            target = {id: val for id, val in schedule_by_time[t].items() if evs[id].remaining_demand - e_del[id] >= self.min_rate}
            if len(target) == 0:
                continue
            sch = OptimizationPostProcessor(self.min_rate, self.cutoff, e_del, const_type=PHASE_AWARE, const_scale=self.const_scale)
            sch.add_evs([evs[id] for id in target])
            sch.obj = sch.post_processing_obj(target)

            sch.compile()
            obj, schedule, runtime = sch.solve()

            for id in schedule:
                rate = project_rate(schedule[id][0])
                processed_schedule[id][t] = rate
                e_del[id] += rate
            # except InfeasibilityException:
            #     print('An error occured at time {0}'.format(t))
        return processed_schedule
