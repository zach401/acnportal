import math
from acnlib.signals.sig_utils import periods_since_midnight, extended_schedule


class Prices:
    def __init__(self, period):
        self.period = period

    def get_prices(self, start, length):
        """
        Get a vector of prices beginning at time start and continuing for length periods.

        :param int start: Time step of the simulation where price vector should begin.
        :param int length: Number of elements in the prices vector. One entry per period.
        :return: vector of floats of length length where each entry is a price which is valid for one period.
        """
        raise NotImplementedError('Prices is an abstract class. It can not be used directly.')

    def get_normalized_demand_charge(self, period, schedule_len):
        """
        Get the demand charge scaled according to the length of the scheduling period.


        :param int period: length of each discrete period in minutes.
        :param int schedule_len: length of the schedule in number of periods.
        :return: float Demand charge scaled for the scheduling period.
        """
        raise NotImplementedError('This class does not implement a demand charge.')


class TOUPrices(Prices):
    def __init__(self, period, init_time, voltage, include_dc=False, revenue=0.3, revenue_max_maginal=False):
        super().__init__(period)
        self.init_time = periods_since_midnight(init_time, self.period)
        if 60 % period == 0:
            period_per_hour = 60 // self.period
            # self.price_schedule = [0.05]*(8*period_per_hour) + [0.12]*(4*period_per_hour) + \
            #                       [0.29]*(6*period_per_hour) + [0.12]*(5*period_per_hour) + \
            #                       [0.05]*period_per_hour  # $/kWh
            self.price_schedule = [0.06]*(8*period_per_hour) + [0.09]*(4*period_per_hour) + \
                                  [0.25]*(6*period_per_hour) + [0.09]*(5*period_per_hour) + \
                                  [0.06]*period_per_hour  # $/kWh
            energy_conversion = (voltage*period)/(1000*60) # #/A*period
            self.price_schedule = [energy_conversion*p for p in self.price_schedule]
        else:
            raise NotImplementedError('Only periods which divide 60 evenly are allowed.')
        if include_dc:
            power_conversion = voltage/1000
            # self.demand_charge = 13.20 # $/kW/month
            self.demand_charge = 15.48  # $/kW/month
            self.demand_charge *= power_conversion # $/A/month
        else:
            self.demand_charge = 0
        if revenue_max_maginal:
            self.revenue = self.demand_charge + max(self.price_schedule) + 1e-6
        else:
            self.revenue = revenue*energy_conversion

    def get_prices(self, start, length):
        """
        Get a vector of prices beginning at time start and continuing for length periods.

        :param int start: Time step of the simulation where price vector should begin.
        :param int length: Number of elements in the prices vector. One entry per period.
        :return: vector of floats of length length where each entry is a price which is valid for one period.
        """

        offset = self.init_time + start
        energy_prices = extended_schedule(self.price_schedule, offset, length)
        return energy_prices

    def get_normalized_demand_charge(self, period, schedule_len):
        """
        Get the demand charge scaled according to the length of the scheduling period.


        :param int period: length of each discrete period in minutes.
        :param int schedule_len: length of the schedule in number of periods.
        :return: float Demand charge scaled for the scheduling period.
        """
        return self.demand_charge * (period*schedule_len)/(30*24*60)


class WinterTOUPrices(TOUPrices):
    def __init__(self, period, init_time, voltage, include_dc=False, revenue=0.3, revenue_max_maginal=False):
        super().__init__(period, init_time, voltage, include_dc, revenue, revenue_max_maginal)
        if 60 % period == 0:
            period_per_hour = 60 // self.period
            self.price_schedule = [0.07]*(8*period_per_hour) + [0.08]*(4*period_per_hour) + \
                                  [0.08]*(6*period_per_hour) + [0.08]*(5*period_per_hour) + \
                                  [0.07]*period_per_hour  # $/kWh
            energy_conversion = (voltage*period)/(1000*60) # #/A*period
            self.price_schedule = [energy_conversion*p for p in self.price_schedule]
        else:
            raise NotImplementedError('Only periods which divide 60 evenly are allowed.')
