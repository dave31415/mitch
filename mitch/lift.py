import numpy as np
from readers import read_merged_messages
from collections import defaultdict
import date_utils as du


def calculate_last_date(sorted_dates, this_date):
    #TODO: add test
    #sorted_dates should be an numpy array of dates
    #uses a quick binary search method from numpy
    index = np.searchsorted(sorted_dates, this_date, side='right')
    if index == 0:
        return None
    return sorted_dates[index-1]


def make_last_date_lookup(messages=None):
    if messages is None:
        print 'reading messages'
        messages = read_merged_messages()
    user_date_set = defaultdict(set)

    for i, message in enumerate(messages):
        user_id = message['user_customer_external_id']
        if not user_id:
            continue
        if 'campaign_sent_at_date' not in message:
            continue
        sent_date_string = message['campaign_sent_at']
        if not sent_date_string:
            continue
        sent_date = du.mdy_to_date(sent_date_string)

        user_date_set[user_id].add(sent_date)

    #make sorted numpy array
    for user_id, date_set in user_date_set.iteritems():
        user_date_set[user_id] = np.array(sorted(list(date_set)))

    return user_date_set


def make_last_date_lookup_function(messages=None):
    # make a closure to return the last message date for a user_id and a date
    # and the number of days since the last message date
    user_date_set = make_last_date_lookup(messages)

    def last_date_function(user_id, this_date):
        if user_id not in user_date_set:
            return None, None

        last_date = calculate_last_date(user_date_set[user_id], this_date)
        if last_date is None:
            return None, None

        days_since_last_date = (this_date - last_date).days
        return last_date, days_since_last_date

    return last_date_function


def count_sales_days_since_message(messages=None, purchase_dates=None):
    if messages is None:
        print 'reading messages'
        messages = read_merged_messages()

    if purchase_dates is None:
        print 'reading purchase dates'
        purchase_dates = get_purchase_dates()

    message_dates = [du.mdy_to_date(i['']) for d in




