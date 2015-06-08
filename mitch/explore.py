from readers import read, read_merged_messages
from collections import defaultdict, Counter
import numpy as np
from matplotlib import pylab as plt
import date_utils as du


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
    for sale in sales:
        user_id = sale['customer_external_id']
        purchase_date_string = sale['purchase_date']
        purchase_date = du.ymd_to_date(purchase_date_string)
        purchase_dates[user_id].append(purchase_date)
    return purchase_dates


def sales_days_from_messages(messages=None, purchase_dates=None,
                             randomize=False, days_max=100):
    if messages is None:
        print 'reading messages'
        messages = read_merged_messages()
    n_messages = len(messages)
    if purchase_dates is None:
        print 'reading purchase dates'
        purchase_dates = get_purchase_dates()

    no_user = 0
    no_date = 0

    count_of_days = Counter()

    for i, message in enumerate(messages):
        if (i % 1000) == 0:
            print "%s of %s, %0.1f percent" % (i, n_messages, 100.0*i/float(n_messages))
        if randomize:
            user_id = message['user_customer_external_id_random']
        else:
            user_id = message['user_customer_external_id']

        if user_id in purchase_dates and user_id:
            user_purchase_dates = purchase_dates[user_id]

            if 'campaign_sent_at' in message and message['campaign_sent_at']:
                date_string = message['campaign_sent_at']
                message_date = du.mdy_to_date(date_string)
                for user_purchase_date in user_purchase_dates:
                    diff_days = (user_purchase_date - message_date).days
                    if abs(diff_days) <= days_max:
                        count_of_days[diff_days] += 1
            else:
                no_date += 1
        else:
            no_user += 1

    print "number with no user in sales: %s of %s" % (no_user, n_messages)
    print "number with no date: %s of %s" % (no_date, n_messages)

    return count_of_days


def days_sales(count_of_days):
    n_days = np.array(count_of_days.keys())
    n_sales = np.array(count_of_days.values())
    s = np.argsort(n_days)
    n_days = n_days[s]
    n_sales = n_sales[s]
    return n_days, n_sales


def plot_days_after_sales(messages=None, purchase_dates=None):

    count_of_days = sales_days_from_messages(messages=messages, purchase_dates=purchase_dates, randomize=False)
    n_days, n_sales = days_sales(count_of_days)

    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(n_days, n_sales, color='blue', marker='o', linestyle='-')
    plt.axvline(x=0, linestyle='--', color='red')
    plt.title('Normal')
    plt.xlabel('N days after message')
    plt.ylabel('N sales')

    count_of_days_randomized = sales_days_from_messages(messages=messages, purchase_dates=purchase_dates, randomize=True)
    n_days_randomized, n_sales_randomized = days_sales(count_of_days_randomized)

    plt.subplot(2, 1, 2)
    plt.plot(n_days_randomized, n_sales_randomized, color='blue', marker='o', linestyle='-')
    plt.axvline(x=0, linestyle='--', color='red')
    plt.title('Users randomized')
    plt.xlabel('N days after message')
    plt.ylabel('N sales')


def plot_days_after_sales_ratio(messages=None, purchase_dates=None):
    #just a single plot with two lines
    count_of_days = sales_days_from_messages(messages=messages, purchase_dates=purchase_dates, randomize=False)
    n_days, n_sales = days_sales(count_of_days)

    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(n_days, n_sales, color='blue', marker='o', linestyle='-', label='Normal')
    plt.axvline(x=0, linestyle='--', color='red')
    plt.xlabel('N days after message')
    plt.ylabel('N sales')

    count_of_days_randomized = sales_days_from_messages(messages=messages, purchase_dates=purchase_dates, randomize=True)
    n_days_randomized, n_sales_randomized = days_sales(count_of_days_randomized)

    plt.plot(n_days_randomized, n_sales_randomized, color='magenta', marker='o',
             linestyle='-', label='Randomized')
    plt.legend()

    plt.subplot(2, 1, 1)
    rat = {}
    for days, count in count_of_days.iteritems():
        if days in count_of_days_randomized:
            count_random = count_of_days_randomized[days]
            if count_random > 0:
                rat[days] = count/float(count_random)
    days = rat.keys()
    ratio = rat.values()

    s = np.argsort(days)
    days = np.array(days)[s]
    ratio = np.array(ratio)[s]

    plt.subplot(2, 1, 2)
    plt.plot(days, ratio, color='blue', marker='o', linestyle='-')
    plt.xlabel('N days after message')
    plt.ylabel('Ratio Normal/Randomized')
    plt.axvline(x=0, linestyle='--', color='red')



