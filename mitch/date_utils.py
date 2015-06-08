from datetime import datetime
#TODO: add tests


def date_string_to_date(date_string, date_format):
    date_string_truncate = date_string.split()
    if len(date_string_truncate) < 1:
        return None
    else:
        return datetime.strptime(date_string_truncate[0], date_format)


def ymd_to_date(date_string):
    date_format = "%Y/%m/%d"
    return date_string_to_date(date_string, date_format)


def mdy_to_date(date_string):
    date_format = "%m/%d/%Y"
    return date_string_to_date(date_string, date_format)

