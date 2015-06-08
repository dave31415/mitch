from datetime import datetime
from memo import memo
#TODO: add tests


def date_string_to_date(date_string, date_format):
    date_string_truncate = date_string.split()
    if len(date_string_truncate) < 1:
        return None
    else:
        return datetime.strptime(date_string_truncate[0], date_format)


@memo
def ymd_to_date(date_string):
    date_format = "%Y/%m/%d"
    return date_string_to_date(date_string, date_format)


@memo
def mdy_to_date(date_string):
    date_format = "%m/%d/%Y"
    return date_string_to_date(date_string, date_format)


def get_all_message_dates(messages):
    #return a sorted and unique list of all message dates in messages
    message_dates = set()
    for message in messages:
        if 'campaign_sent_at' not in message:
            continue
        if not message['campaign_sent_at']:
            continue
        message_date = mdy_to_date(message['campaign_sent_at'])
        message_dates.add(message_date)
    return sorted(list(message_dates))


def get_all_purchase_dates(purchases):
    #return a sorted and unique list of all message dates in messages
    purchase_dates = set()
    for purchase in purchases:
        if 'purchase_date' not in purchase:
            continue
        if not purchase['purchase_date']:
            continue
        purchase_date = ymd_to_date(purchase['purchase_date'])
        purchase_dates.add(purchase_date)
    return sorted(list(purchase_dates))