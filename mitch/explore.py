from readers import read, read_merged_messages
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np
from matplotlib import pylab as plt


def sales_versus_messages_plot():
    sales = read('sales')

    messages = read_merged_messages()
    user_data = {}

    for sale in sales:
        user_id = sale['customer_external_id']
        if user_id not in user_data:
            user_data[user_id] = {'n_sales': 0, 'n_messages': 0}

        user_data[user_id]['n_sales'] += 1

    for message in messages:
        user_id = message['user_customer_external_id']
        if user_id not in user_data:
            user_data[user_id] = {'n_sales': 0, 'n_messages': 0}

        user_data[user_id]['n_messages'] += 1

    n_messages = [v['n_messages'] for v in user_data.itervalues()]
    n_sales = [v['n_sales'] for v in user_data.itervalues()]

    return user_data, n_messages, n_sales


def get_purchase_dates():
    sales = read('sales')
    purchase_dates = defaultdict(list)
    date_format = "%Y/%m/%d"
    for sale in sales:
        user_id = sale['customer_external_id']
        purchase_date_string = sale['purchase_date']
        purchase_date = datetime.strptime(purchase_date_string, date_format)
        purchase_dates[user_id].append(purchase_date)
    return purchase_dates


def sales_days_from_messages(messages=None, purchase_dates=None, randomize=False):
    date_format = "%m/%d/%Y %H:%M:%S"
    if messages is None:
        messages = read_merged_messages()
    n_messages = len(messages)
    if purchase_dates is None:
        purchase_dates = get_purchase_dates()
    days_between_purchase_and_message = []

    no_user = 0
    no_date = 0

    for i, message in enumerate(messages):
        if (i % 1000) == 0:
            print "%s of %s, %0.1f percent" % (i, n_messages, 100.0*i/float(n_messages))
        if randomize:
            user_id = message['user_customer_external_id_random']
        else:
            user_id = message['user_customer_external_id']

        if user_id in purchase_dates:
            user_purchase_dates = purchase_dates[user_id]

            if 'campaign_sent_at' in message and message['campaign_sent_at']:
                date_string = message['campaign_sent_at']
                message_date = datetime.strptime(date_string, date_format)
                for user_purchase_date in user_purchase_dates:
                    diff_days = (user_purchase_date - message_date).days
                    days_between_purchase_and_message.append(diff_days)
            else:
                no_date += 1
        else:
            no_user += 1

    print "number with no user in sales: %s of %s" % (no_user, n_messages)
    print "number with no date: %s of %s" % (no_date, n_messages)

    return days_between_purchase_and_message


def count_days(days):
    count_of_days = Counter()
    for d in days:
        count_of_days[d] += 1

    n_days = np.array(count_of_days.keys())
    n_sales = np.array(count_of_days.values())
    s = np.argsort(n_days)
    n_days = n_days[s]
    n_sales = n_sales[s]
    return n_days, n_sales


def plot_days_after_sales(messages=None, purchase_dates=None):

    days = sales_days_from_messages(messages=None, purchase_dates=None, randomize=False)
    n_days, n_sales = count_days(days)

    plt.clf()
    plt.subplots(121)
    plt.plot(n_days, n_sales, 'bo')
    plt.axvline(x=0, linestyle='--', color='red')
    plt.title('')
    plt.xlabel('N days after message')
    plt.ylabel('N sales')

    days_randomized = sales_days_from_messages(messages=None, purchase_dates=None, randomize=True)
    n_days_randomized, n_sales_randomized = count_days(days_randomized)

    plt.subplots(121)
    plt.plot(n_days_randomized, n_sales_randomized, 'bo')
    plt.axvline(x=0, linestyle='--', color='red')
    plt.title('Users randomized')
    plt.xlabel('N days after message')
    plt.ylabel('N sales')


