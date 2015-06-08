import numpy as np
from readers import read_merged_messages
from collections import defaultdict
import date_utils as du
from datetime import datetime, timedelta
from readers import read
from collections import Counter
from matplotlib import pylab as plt


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


def count_sales_days_since_message(messages=None, purchases=None):
    buffer_days = 60
    delta_days = timedelta(buffer_days)

    if messages is None:
        print 'reading messages'
        messages = read_merged_messages()

    if purchases is None:
        print 'reading purchase dates'
        purchases = list(read('sales'))

    message_dates = du.get_all_message_dates(messages)
    message_date_min = min(message_dates)
    #start at a date with enough time for the first messages to effect things
    start_date = message_date_min + delta_days
    end_date = max(message_dates)

    n_dates = (end_date - start_date).days + 1
    print 'start_date:', start_date
    print 'end_date:', end_date
    print 'n_dates: %s\n' % n_dates

    purchase_dates_unique = du.get_all_purchase_dates(purchases)
    purchase_dates_unique = [i for i in purchase_dates_unique if start_date <= i <= end_date]
    n_purchase_dates_unique = len(purchase_dates_unique)

    print 'purchase_start_date:', min(purchase_dates_unique)
    print 'purchase_end_date:', max(purchase_dates_unique)
    print 'n_purchase_dates_unique:', n_purchase_dates_unique

    print 'making last date lookup function'

    last_date_function = make_last_date_lookup_function(messages)

    n_days_counter = Counter()
    n_days_counter_random = Counter()

    purchase_dates = [du.ymd_to_date(i['purchase_date']) for i in purchases]
    n_purchase_dates = len(purchase_dates)

    for purchase in purchases:
        if 'purchase_date' not in purchase:
            continue

        purchase_date_string = purchase['purchase_date']
        if not purchase_date_string:
            continue

        if 'customer_external_id' not in purchase:
            continue

        user_id = purchase['customer_external_id']

        if not user_id:
            continue

        purchase_date = du.ymd_to_date(purchase_date_string)

        last_message_date, n_days_since_last_message = \
            last_date_function(user_id, purchase_date)

        random_purchase_date_index = np.random.randint(n_purchase_dates)
        random_purchase_date = purchase_dates[random_purchase_date_index]

        last_message_date_random, n_days_since_last_message_random = \
            last_date_function(user_id, random_purchase_date)

        n_days_counter[n_days_since_last_message] += 1
        n_days_counter_random[n_days_since_last_message_random] += 1

    return n_days_counter, n_days_counter_random


def plot_ratios(n_days_counter, n_days_counter_random,scale=1.0):
    n_days = [n for n in n_days_counter.keys() if n < 100]

    data = [(n, n_days_counter[n], n_days_counter_random[n],
             (1.0 + n_days_counter[n])/(1.0 + n_days_counter_random[n]))
            for n in n_days if n is not None]

    days = np.array([d[0] for d in data])
    ratio = np.array([d[3] for d in data])*scale

    alpha = 0.35
    beta = 10.0
    baseline = 0.9

    print days
    model = baseline + alpha*np.exp(-days/beta)

    plt.plot(days, ratio, color='blue')
    plt.plot(days, baseline+days*0.0, color='gray', linestyle='--')
    plt.plot(days, model, color='red')
    plt.xlabel('N days since message')
    plt.ylabel('Sales Lift')



