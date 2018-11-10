from datetime import datetime
import math

def _periods_since_midnight(t, period):
    midnight = t.replace(hour=0, minute=0, second=0, microsecond=0)
    return int((t - midnight).total_seconds() / period)


def _extended_schedule(schedule, start, length):
    tile_multiple = math.ceil((start + length)/len(schedule))
    tiled_schedule = schedule*tile_multiple
    return tiled_schedule[start:start+length]


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


class TOUPrices(Prices):
    def __init__(self, period, init_time, energy_conversion):
        super().__init__(period)
        self.init_time = _periods_since_midnight(init_time, self.period)
        if 60 % period == 0:
            period_per_hour = 60 // self.period
            self.price_schedule = [0.05]*(8*period_per_hour) + [0.12]*(4*period_per_hour) + \
                                  [0.29]*(6*period_per_hour) + [0.12]*(5*period_per_hour) + \
                                  [0.05]*period_per_hour  # $/kWh
            self.price_schedule = [energy_conversion*p for p in self.price_schedule]
        else:
            raise NotImplementedError('Only periods which divide 60 evenly are allowed.')
        self.demand_charge = 13.20 * energy_conversion  # $/kW/month
        self.revenue = 0.30*energy_conversion

    def get_prices(self, start, length):
        """
        Get a vector of prices beginning at time start and continuing for length periods.

        :param int start: Time step of the simulation where price vector should begin.
        :param int length: Number of elements in the prices vector. One entry per period.
        :return: vector of floats of length length where each entry is a price which is valid for one period.
        """

        offset = self.init_time + start
        energy_prices = _extended_schedule(self.price_schedule, offset, length)
        return energy_prices
