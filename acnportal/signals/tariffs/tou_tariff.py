import json
import os.path
from datetime import datetime, timedelta


class TimeOfUseTariff(object):
    """ Class to represent the time-of-use tariff schedule of a utility.

    Args:
        filename (str): Name of the tariff structure file excluding its extension which should be .json.
        tariff_dir (str): Path to the directory where tariff files are stored. Defaults to the pre-installed tariff
            directory included with the package.

    Attributes:
        name (str): Name of the tariff.
        effective (str): Date when the tariff when into effect. Useful if a tariff has been modified but kept the same
            name.
    """
    def __init__(self, filename, tariff_dir='tariff_schedules'):
        with open(os.path.join(os.path.dirname(__file__), tariff_dir, filename + '.json')) as f:
            tariff_file = json.load(f)
        self.name = tariff_file['name']
        self.effective = tariff_file['effective']
        self._schedule = [TariffSchedule(s) for s in tariff_file['schedule']]
        self._schedule.sort(key=lambda x: (x.start_month, x.start_day))

    def _get_tariff_schedule(self, dt):
        """ Return the tariff schedule in effect for the given datetime.

        Args:
            dt (datetime): A datetime object.

        Returns:
            TariffSchedule: An object representing the tariff schedule for a given time.
        """
        valid_schedules = [s for s in self._schedule if s.dow_mask[dt.weekday()]]
        for i, s in enumerate(valid_schedules):
            if (dt.month, dt.day) <= s.month_day and s.dow_mask[dt.weekday()]:
                return valid_schedules[i-1]
        return valid_schedules[-1]

    def get_tariff(self, dt):
        """ Return the tariff in effect at a given datetime.

        Args:
            dt (datetime): A datetime object.

        Returns:
            float: Tariff in $/kWh
        """
        tariff_schedule = self._get_tariff_schedule(dt)
        target_hour = dt.hour # + dt.minute/60 + dt.second/3600
        for r in sorted(tariff_schedule.tariffs, reverse=True):
            if target_hour >= r[0]:
                return r[1]
        return tariff_schedule.tariffs[-1][1]

    def get_tariffs(self, start, length, period):
        """ Return a list of tariffs beginning at time start and continuing for length time intervals.


        Args:
            start (datetime): Initial time when the tariff list should begin.
            length (int): Number of time intervals to cover.
            period (int): Length of each time interval in minutes.

        Returns:
            List[float]: A list of tariffs, one for each time interval.
        """
        return [self.get_tariff(start + t*timedelta(minutes=period)) for t in range(length)]

    def get_demand_charge(self, dt):
        tariff_schedule = self._get_tariff_schedule(dt)
        return tariff_schedule.demand_charge


class TariffSchedule(object):
    """ Container to store individual tariff schedules.

    Args:
        doc (dict): A tariff schedule json document as a dict.
    Attributes:
        id (str): Identifier of the schedule.
        start_month (int): Month when the schedule takes effect.
        start_day (int): Day of the month when the schedule takes effect.
        dow_mask (List[bool]): Boolean mask which represents which days of the week a schedule is valid for. 0 index is
            Monday, 6 is Sunday.
        tariffs (List[Tuple[float, float]]): List of time, tariff pairs where time is measured in hours since midnight
            and tariffs are in $/kWh.
    """
    def __init__(self, doc):
        self.id = doc['id']
        self.start_month = int(doc['effective_start'].split('-')[0])
        self.start_day = int(doc['effective_start'].split('-')[1])
        if doc['dow_mask'] == 'WEEKDAYS':
            self.dow_mask = [True] * 5 + [False] * 2
        elif doc['dow_mask'] == 'WEEKENDS':
            self.dow_mask = [False] * 5 + [True] * 2
        elif doc['dow_mask'] == 'ALL':
            self.dow_mask = [True] * 7
        else:
            raise ValueError('dow_mask must be WEEKEDAY, WEEKENDS, or ALL.')
        self.tariffs = [(int(doc['times'][i]), float(doc['tariffs'][i])) for i in range(len(doc['times']))]
        self.tariffs.sort()
        self.demand_charge = doc['demand_charge']

    @property
    def month_day(self):
        """ Return a tuple of (start_month, start_day)."""
        return self.start_month, self.start_day
