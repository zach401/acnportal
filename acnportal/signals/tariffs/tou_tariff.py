import json
import os.path
from datetime import datetime, timedelta
from decimal import Decimal
from copy import copy
from typing import List


class TariffSchedule(object):
    """ Container to store individual tariff schedules.

    Args:
        doc (dict): A tariff schedule json document as a dict.
    Attributes:
        id (str): Identifier of the schedule.
        start (int): (Month, Day) when the schedule takes effect.
        end (int): (Month, Day) of the last day when the schedule is in effect.
        dow_mask (List[bool]): Boolean mask which represents which days of the week a schedule is valid for. 0 index is
            Monday, 6 is Sunday.
        tariffs (List[Tuple[float, float]]): List of time, tariff pairs where time is measured in hours since midnight
            and tariffs are in $/kWh.
    """

    def __init__(self, doc):
        self.id = doc["id"]
        self.start = tuple(int(x) for x in doc["effective_start"].split("-"))
        self.end = tuple(int(x) for x in doc["effective_end"].split("-"))
        if doc["dow_mask"] == "WEEKDAYS":
            self.dow_mask = [True] * 5 + [False] * 2
        elif doc["dow_mask"] == "WEEKENDS":
            self.dow_mask = [False] * 5 + [True] * 2
        elif doc["dow_mask"] == "ALL":
            self.dow_mask = [True] * 7
        else:
            raise ValueError("dow_mask must be WEEKEDAY, WEEKENDS, or ALL.")
        self.tariffs = [
            (Decimal(doc["times"][i]), float(doc["tariffs"][i]))
            for i in range(len(doc["times"]))
        ]
        self.tariffs.sort()
        if self.tariffs[0][0] != 0:
            raise ValueError(
                "Error with tariff schedule. "
                "Schedule must start at time 0, began with time {0}".format(
                    self.tariffs[0][0]
                )
            )
        self.demand_charge = doc["demand_charge"]


class TimeOfUseTariff(object):
    """ Class to represent the time-of-use tariff schedule of a utility.

    Args:
        filename (str): Name of the tariff structure file excluding its extension which should be .json.
        tariff_dir (str): Path to the directory where tariff files are stored. Defaults to the pre-installed tariff
            directory included with the package.

    Attributes:
        name (str): Name of the tariff.
        effective (str): Date when the tariff went into effect. Useful if a tariff has been modified but kept the same
            name.
    """

    def __init__(self, filename, tariff_dir="tariff_schedules"):
        with open(
            os.path.join(os.path.dirname(__file__), tariff_dir, filename + ".json")
        ) as f:
            tariff_file = json.load(f)
        self.name = tariff_file["name"]
        self.effective = tariff_file["effective"]
        self._schedule = [TariffSchedule(s) for s in tariff_file["schedule"]]
        to_add = []
        for s in self._schedule:
            if s.end < s.start:
                s_copy = copy(s)
                s_copy.start = (1, 1)
                to_add.append(s_copy)
                s.end = (12, 31)
        self._schedule.extend(to_add)
        self._schedule.sort(key=lambda x: x.start)

    def _get_tariff_schedule(self, date_time: datetime) -> TariffSchedule:
        """ Return the tariff schedule in effect for the given datetime.

        Args:
            date_time (datetime): A datetime object.

        Returns:
            TariffSchedule: An object representing the tariff schedule for a given time.
        """
        valid_schedules = [
            s
            for s in self._schedule
            if s.dow_mask[date_time.weekday()]
            and s.start <= (date_time.month, date_time.day) <= s.end
        ]
        if len(valid_schedules) == 0:
            raise ValueError("No valid tariff schedule for {0}".format(date_time))
        elif len(valid_schedules) > 1:
            raise ValueError(
                "More than one tariff schedule is valid for {0}".format(date_time)
            )
        else:
            return valid_schedules[0]

    def get_tariff(self, date_time: datetime) -> float:
        """ Return the tariff in effect at a given datetime.

        Args:
            date_time (datetime): A datetime object.

        Returns:
            float: Tariff [$/kWh]
        """
        tariff_schedule = self._get_tariff_schedule(date_time)
        target_hour = (
            Decimal(date_time.hour)
            + Decimal(date_time.minute) / 60
            + Decimal(date_time.second) / 3600
        )
        for r in sorted(tariff_schedule.tariffs, reverse=True):
            if target_hour >= r[0]:
                return r[1]
        raise ValueError(
            "Error with tariff schedule. Could not find a valid price for {0}.".format(
                date_time
            )
        )

    def get_tariffs(self, start: datetime, length: int, period: int) -> List[float]:
        """ Return a list of tariffs beginning at time start and continuing for length time intervals.


        Args:
            start (datetime): Initial time when the tariff list should begin.
            length (int): Number of time intervals to cover.
            period (int): Length of each time interval in minutes.

        Returns:
            List[float]: A list of tariffs, one for each time interval. [$/kWh]
        """
        return [
            self.get_tariff(start + t * timedelta(minutes=period))
            for t in range(length)
        ]

    def get_demand_charge(self, date_time: datetime) -> float:
        """ Return the demand charge for rates at the given time.

        Args:
            date_time (datetime): A datetime object.

        Returns:
            float: Tariff [$/kW]
        """
        tariff_schedule = self._get_tariff_schedule(date_time)
        return tariff_schedule.demand_charge
