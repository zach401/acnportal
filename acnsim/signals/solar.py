from signals.sig_utils import *
import pandas as pd
import numpy as np

class Solar:
    def __init__(self, period, init_time):
        self.period = period
        self.init_time = periods_since_midnight(init_time, self.period)

    def get_generation(self, start, length):
        """
        Get a vector of solar generation beginning at time start and continuing for length periods.

        :param int start: Time step of the simulation where price vector should begin.
        :param int length: Number of elements in the prices vector. One entry per period.
        :return: vector of floats of length length where each entry is a price which is valid for one period.
        """
        raise NotImplementedError('Solar is an abstract class. It can not be used directly.')


class SolarCSV(Solar):
    def __init__(self, period, init_time, csv_source, scale=None, capacity=None, index_col='Time', col_name='Average Last 5 points'):
        super().__init__(period, init_time)
        self.csv = csv_source
        raw_solar = pd.read_csv(self.csv, index_col=index_col, infer_datetime_format=True, parse_dates=True)
        raw_solar = raw_solar.resample('{0}T'.format(self.period)).pad()
        self.scale=scale
        gen = raw_solar[col_name].fillna(0).clip(lower=0).values
        if self.scale is not None:
            if capacity is None:
                gen = (gen / np.max(gen))*self.scale
            else:
                gen = (gen / capacity) * self.scale
        self.generation = list(gen)


    def get_generation(self, start, length):
        """
        Get a vector of prices beginning at time start and continuing for length periods.

        :param int start: Time step of the simulation where price vector should begin.
        :param int length: Number of elements in the prices vector. One entry per period.
        :return: vector of floats of length length where each entry is a price which is valid for one period.
        """

        offset = self.init_time + start
        gen = extended_schedule(self.generation, offset, length)
        return gen


