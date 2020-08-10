from datetime import datetime
import pytz


def http_date(dt):
    """ Convert datetime object into http date according to RFC 1123.

    :param datetime dt: datetime object to convert
    :return: dt as a string according to RFC 1123 format
    :rtype: str
    """
    return dt.astimezone(pytz.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")


def parse_http_date(ds, tz):
    """ Convert a string in RFC 1123 format to a datetime object

    :param str ds: string representing a datetime in RFC 1123 format.
    :param pytz.timezone tz: timezone to convert the string to as a pytz object.
    :return: datetime object.
    :rtype: datetime
    """
    dt = pytz.UTC.localize(datetime.strptime(ds, "%a, %d %b %Y %H:%M:%S GMT"))
    return dt.astimezone(tz)


def parse_dates(doc):
    """ Convert all datetime fields in RFC 1123 format in doc to datetime objects.

    :param dict doc: document to be converted as a dictionary.
    :return: doc with all RFC 1123 datetime fields replaced by datetime objects.
    :rtype: dict
    """
    tz = pytz.timezone(doc["timezone"])
    for field in doc:
        if isinstance(doc[field], str):
            try:
                dt = parse_http_date(doc[field], tz)
                doc[field] = dt
            except ValueError:
                pass
        if isinstance(doc[field], dict) and "timestamps" in doc[field]:
            doc[field]["timestamps"] = [
                parse_http_date(ds, tz) for ds in doc[field]["timestamps"]
            ]
