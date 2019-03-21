import requests
from datetime import datetime
from .utils import parse_dates, http_date


class C2Client(object):
    """ API client for the Caltech Charging (C2) Dataset.

    Args:
        api_token (str): API token needed to access the Caltech Charging (C2) API.
        url (str): Base url for all API calls. Defaults to the standard C2 API url.

    Attributes:
        token (str): See api_token in Args.
        url (str): See url in Args.

    """

    def __init__(self, api_token, url='http://131.215.140.211/c2data/v1/'):
        self.token = api_token
        self.url = url

    def get_sessions(self, site, cond=None):
        """ Generator to return sessions from the C2 dataset one at a time.

        Args:
            site (str): ACN ID from which data should be gathered.
            token (str): API token needed to access the Caltech Charging (C2) API.
            cond (str): String of conditions. See API reference for the where parameter.

        Yields:
            Dict: Session as a dictionary.

        Raises:
            ValueError: Raised if the site name is not valid.
        """
        # TODO: (zach) Use API to autodetect if a new site is added.
        if site not in {'caltech', 'jpl'}:
            raise ValueError("Invalid site name. Must be either 'caltech' or 'jpl'")

        endpoint = 'sessions/' + site
        conditions = '?where={0}'.format(cond) if cond is not None else ''
        sort = '&sort=connectionTime'
        r = requests.get(self.url + endpoint + conditions + sort, auth=(self.token, ''))
        payload = r.json()
        while True:
            for s in payload['_items']:
                parse_dates(s)
                yield s
            if 'next' in payload['_links']:
                r = requests.get(self.url + payload['_links']['next']['href'], auth=(self.token, ''))
                payload = r.json()
            else:
                break

    def get_sessions_by_time(self, site, start, end, min_energy=0):
        """ Wrapper for get_sessions with condition based on start and end times and a minimum energy delivered.

        Args:
            start (datetime): Only return sessions which began after start.
            end (datetime): Only return session which began before end.
            min_energy (float): Only return sessions where the kWhDelivered is greater than or equal to min_energy.

        Yields:
            See get_sessions.

        Raises:
            See get_sessions.
        """
        start_str = http_date(start)
        end_str = http_date(end)
        cond = 'connectionTime >= "{0}" and connectionTime <= "{1}" and kWhDelivered > {2}'.format(start_str,
                                                                                                   end_str,
                                                                                                   min_energy)
        return self.get_sessions(site, cond)
