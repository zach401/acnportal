from datetime import datetime

import pytz
import requests

BASE_URL = 'http://131.215.140.211/c2data/v1/'


def _http_date(dt):
    """ Convert datetime object into http date according to RFC 1123.

    :param datetime dt: datetime object to convert
    :return: dt as a string according to RFC 1123 format
    :rtype: str
    """
    return dt.astimezone(pytz.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')


def _parse_http_date(ds, tz):
    """ Convert a string in RFC 1123 format to a datetime object

    :param str ds: string representing a datetime in RFC 1123 format.
    :param pytz.timezone tz: timezone to convert the string to as a pytz object.
    :return: datetime object.
    :rtype: datetime
    """
    return datetime.strptime(ds, '%a, %d %b %Y %H:%M:%S GMT').astimezone(tz)


def _parse_dates(doc):
    """ Convert all datetime fields in RFC 1123 format in doc to datetime objects.

    :param dict doc: document to be converted as a dictionary.
    :return: doc with all RFC 1123 datetime fields replaced by datetime objects.
    :rtype: dict
    """
    tz = pytz.timezone(doc['timezone'])
    for field in doc:
        if isinstance(doc[field], str):
            try:
                dt = _parse_http_date(doc[field], tz)
                doc[field] = dt
            except ValueError:
                pass


def get_sessions(token, cond=None):
    """ Generator to return sessions from the C2 dataset one at a time.

    :param str token: API token needed to access the API.
    :param str cond: string of conditions. See API reference for where parameter.
    :return: Session as a dictionary. (yields)
    :rtype: dict
    """
    endpoint = 'sessions/caltech'
    conditions = '?where={0}'.format(cond) if cond is not None else ''
    sort = '&sort=connectionTime'
    r = requests.get(BASE_URL + endpoint + conditions + sort, auth=(token, ''))
    payload = r.json()
    while True:
        for s in payload['_items']:
            _parse_dates(s)
            yield s
        if 'next' in payload['_links']:
            r = requests.get(BASE_URL + payload['_links']['next']['href'], auth=(token, ''))
            payload = r.json()
        else:
            break


def get_sessions_by_time(token, start, end, min_energy=0):
    """ Wrapper for get_sessions with condition based on start and end times and a minimum energy delivered.

    :param token: API token needed to access the API.
    :param start: Only return sessions which began after start
    :param end: Only return sessions which began before end
    :param min_energy: Only return sessions which delivered at least min_energy kWhs.
    :return: Session as a dictionary. (yields)
    :rtype: dict
    """
    start_str = _http_date(start)
    end_str = _http_date(end)
    cond = 'connectionTime >= "{0}" and connectionTime <= "{1}" and kWhDelivered > {2}'.format(start_str, end_str,
                                                                                               min_energy)
    return get_sessions(token, cond)
