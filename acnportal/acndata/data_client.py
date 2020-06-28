from typing import Optional

import requests
from datetime import datetime
from .utils import parse_dates, http_date


class DataClient(object):
    """ API client for acndata.

    Args:
        api_token (str): API token needed to access the acndata API.
        url (str): Base url for all API calls. Defaults to the standard acndata API url.

    Attributes:
        token (str): See api_token in Args.
        url (str): See url in Args.

    """

    def __init__(self, api_token, url="https://ev.caltech.edu/api/v1/"):
        self.token = api_token
        self.url = url

    def get_sessions(self, site, cond=None, project=None, sort=None, timeseries=False):
        """ Generator to return sessions from the acndata dataset one at a time.

        Args:
            site (str): ACN ID from which data should be gathered.
            cond (str): String of conditions. See API reference for the where parameter.

        Yields:
            Dict: Session as a dictionary.

        Raises:
            ValueError: Raised if the site name is not valid.
        """
        # TODO: (zach) Use API to autodetect if a new site is added.
        if site not in {"caltech", "jpl", "office001"}:
            raise ValueError(
                "Invalid site name. Must be either 'caltech', 'jpl', or 'office001'."
            )

        limit = 100
        endpoint = "sessions/" + site
        if timeseries:
            endpoint += "/ts/"
            limit = 1
        args = []
        if cond is not None:
            args.append("where={0}".format(cond))
        if project is not None:
            args.append("project={0}".format(project))
        if sort is not None:
            args.append("sort={0}".format(sort))
        args.append("max_results={0}".format(limit))
        query_string = "?" + "&".join(args) if len(args) > 0 else ""
        r = requests.get(self.url + endpoint + query_string, auth=(self.token, ""))
        payload = r.json()
        while True:
            for s in payload["_items"]:
                parse_dates(s)
                yield s
            if "next" in payload["_links"]:
                r = requests.get(
                    self.url + payload["_links"]["next"]["href"], auth=(self.token, "")
                )
                payload = r.json()
            else:
                break

    def count_sessions(self, site, cond=None):
        """ Return the number of sessions which match the given query

        Args:
            site (str): ACN ID from which data should be gathered.
            cond (str): String of conditions. See API reference for the where parameter.

        Returns:
            int: Number of sessions which match the query.

        Raises:
            ValueError: Raised if the site name is not valid.
        """
        if site not in {"caltech", "jpl", "office001"}:
            raise ValueError("Invalid site name. Must be either 'caltech' or 'jpl'")

        endpoint = "sessions/" + site
        args = []
        if cond is not None:
            args.append("where={0}".format(cond))
        args.append("limit=1")
        query_string = "?" + "&".join(args) if len(args) > 0 else ""
        auth_header = {"Authorization": "Bearer {0}".format(self.token)}
        r = requests.head(self.url + endpoint + query_string, headers=auth_header)
        return r.headers["x-total-count"]

    def get_sessions_by_time(
        self,
        site,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        min_energy=None,
        timeseries=False,
        count=False,
    ):
        """ Wrapper for get_sessions with condition based on start and end times and a minimum energy delivered.

        Args:
            site (str): Site where data should be gathered.
            start (datetime): Only return sessions which began after start.
            end (datetime): Only return session which began before end.
            min_energy (float): Only return sessions where the kWhDelivered is greater than or equal to min_energy.
            timeseries (bool): If True return the time-series of charging rates and pilot signals. Default False.
            count (bool): If True return the number of sessions which would be returned by the function. Default False.

        Yields:
            If count is False see get_sessions else see count_sessions.

        Raises:
            See get_sessions/count_sessions.
        """
        cond = []
        if start is not None:
            cond.append('connectionTime >= "{0}"'.format(http_date(start)))
        if end is not None:
            cond.append('connectionTime <= "{0}"'.format(http_date(end)))
        if min_energy is not None:
            cond.append("kWhDelivered > {0}".format(min_energy))
        condition = " and ".join(cond)
        if count:
            return self.count_sessions(site, condition)
        else:
            return self.get_sessions(
                site, condition, sort="connectionTime", timeseries=timeseries
            )
