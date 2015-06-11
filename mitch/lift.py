import numpy as np
from readers import read_merged_messages
from collections import defaultdict
import date_utils as du
from datetime import timedelta
from readers import read
from matplotlib import pylab as plt
import time
from fitting import boot_fit, fit_ratio, exponential_lift, exponential_with_baseline


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
        if 'campaign_sent_at' not in message:
            continue
        sent_date_string = message['campaign_sent_at']
        if not sent_date_string:
            continue
        sent_date = du.mdy_to_date(sent_date_string)

        user_date_set[user_id].add(sent_date)

    #make sorted numpy array
    for user_id, date_set in user_date_set.iteritems():
        user_date_set[user_id] = np.array(sorted(list(date_set)))

    print '%s users in lookup' % len(user_date_set)

    return user_date_set


def make_last_date_lookup_function(messages=None):
    # make a closure to return the last message date for a user_id and a date
    # and the number of days since that last message date
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


def get_start_end_dates(messages, buffer_days):
    delta_days = timedelta(buffer_days)
    message_dates = du.get_all_message_dates(messages)
    message_date_min = min(message_dates)
    #start at a date with enough time for the first messages to effect things
    start_date = message_date_min + delta_days
    end_date = max(message_dates)
    return start_date, end_date


def purchase_date_user_id(purchase):
    if 'purchase_date' not in purchase:
        return None, None

    purchase_date_string = purchase['purchase_date']
    if not purchase_date_string:
        return None, None

    if 'customer_external_id' not in purchase:
        return None, None

    user_id = purchase['customer_external_id']

    if not user_id:
        return None, None

    purchase_date = du.ymd_to_date(purchase_date_string)

    return purchase_date, user_id


def messages_purchases(messages, purchases):
    #utility for optionally reading these
    if messages is None:
        print 'reading messages'
        messages = read_merged_messages()

    if purchases is None:
        print 'reading purchase dates'
        purchases = read('sales')

    return messages, purchases


def count_sales_days_since_message(messages=None, purchases=None,
                                   niter_random=1, doplot=False, color="blue"):
    buffer_days = 60

    start_time = time.time()

    #read if not provided
    messages, purchases = messages_purchases(messages, purchases)

    #be sure it's a list not a generator
    if not isinstance(purchases, list):
        print 'Warning - converting purchases from a generator to a list'
        purchases = list(purchases)

    start_date, end_date = get_start_end_dates(messages, buffer_days)

    n_dates = (end_date - start_date).days + 1
    print 'start_date:', start_date
    print 'end_date:', end_date
    print 'n_dates: %s\n' % n_dates

    print 'making last date lookup function'

    last_date_function = make_last_date_lookup_function(messages)

    # to calculate P(n | X=1), X is sale
    n_days_counter = defaultdict(float)
    # to calculate P(n)
    n_days_counter_random = defaultdict(float)

    purchase_dates = [du.ymd_to_date(i['purchase_date']) for i in purchases]
    n_purchase_dates = len(purchase_dates)

    # make a pseudo-random seed that is still deterministic
    seed = hash(purchase_dates.__repr__()) % 100000
    np.random.seed(seed)

    n_purchases = len(purchases)

    for i, purchase in enumerate(purchases):
        if i % 10000 == 0:
            print '%s of %s' % (i, n_purchases)

        purchase_date, user_id = purchase_date_user_id(purchase)
        if purchase_date is None:
            continue

        # calculate P(n | X=1), X=1 is sale happens
        last_message_date, n_days_since_last_message = \
            last_date_function(user_id, purchase_date)

        n_days_counter[n_days_since_last_message] += 1

        # calculate P(n) can do this simply by choosing a random purchase date
        # rather than the actual purchase date

        for iter in xrange(niter_random):
            random_purchase_date_index = np.random.randint(n_purchase_dates)
            random_purchase_date = purchase_dates[random_purchase_date_index]

            last_message_date_random, n_days_since_last_message_random = \
                last_date_function(user_id, random_purchase_date)

            #normalize on the fly by dividing by niter_random
            n_days_counter_random[n_days_since_last_message_random] += 1.0/float(niter_random)

    np.random.seed(None)

    end_time = time.time()
    run_time = end_time - start_time
    print "Runtime: %s seconds for %s purchases and niter_random=%s" % (int(run_time), n_purchases, niter_random)

    if doplot:
        plot_ratios(n_days_counter, n_days_counter_random, scale=1.0, color=color)

    return n_days_counter, n_days_counter_random


def get_days_ratio(n_days_counter, n_days_counter_random, scale=1.0, nmax=200):
    n_days = [n for n in n_days_counter.keys() if n < nmax]

    data = [(n, n_days_counter[n], n_days_counter_random[n],
             (1.0 + n_days_counter[n])/(1.0 + n_days_counter_random[n]))
            for n in n_days if n is not None]

    days = np.array([d[0] for d in data])
    ratio = np.array([d[3] for d in data])*scale
    return days, ratio


def binning_function(days, factor):
    gamma = 0.4
    scaled = (days*factor)**(gamma)
    return scaled.astype(int)


def bin_days_ratio(days, ratio, factor=9.0):
    #TODO: maybe refactor and use itertools
    bins = binning_function(days, factor)
    data = {}

    for day, rat, bin in zip(days, ratio, bins):
        if bin not in data:
            data[bin] = {'num': 0, 'sum_days': 0.0, 'sum_ratio': 0.0, 'sum_ratio_squared': 0.0}
        data[bin]['num'] += 1
        data[bin]['sum_days'] += day
        data[bin]['sum_ratio'] += rat
        data[bin]['sum_ratio_squared'] += rat*rat
    for bin, vals in data.iteritems():
        vals['mean_days'] = vals['sum_days']/vals['num']
        vals['mean_ratio'] = vals['sum_ratio']/vals['num']
        vals['sigma_ratio'] = np.sqrt((vals['sum_ratio_squared']/vals['num']) - vals['mean_ratio']**2)
    return data


def get_binned_days_ratio_from_counters(days, ratio, factor=9.0):
    binned_data = bin_days_ratio(days, ratio, factor=factor)
    mean_days = np.array([i['mean_days'] for i in binned_data.values()])
    mean_ratio = np.array([i['mean_ratio'] for i in binned_data.values()])
    return mean_days, mean_ratio


def get_lift_data(messages=None, purchases=None):
    messages, purchases = messages_purchases(messages, purchases)
    n_days_counter, n_days_counter_random = \
        count_sales_days_since_message(messages=messages, purchases=purchases)
    days, ratio = get_days_ratio(n_days_counter, n_days_counter_random, nmax=200)
    mean_days, mean_ratio = get_binned_days_ratio_from_counters(days, ratio)

    return days, ratio, mean_days, mean_ratio


def plot_ratios(messages=None, purchases=None, title=''):
    days, ratio, mean_days, mean_ratio = get_lift_data(messages, purchases)

    plt.plot(days, ratio, color='gray', alpha=0.3)
    plt.xlabel('N days since message')
    plt.ylabel('Sales Lift')

    plt.plot(mean_days, mean_ratio, color="blue", marker='o', markersize=8)

    fit, boot = boot_fit(mean_days, mean_ratio, nboot=1000)
    baseline = fit['baseline']
    alpha = fit['alpha']
    beta = fit['beta']

    lift30 = exponential_lift(alpha, beta)
    alpha_error = fit['alpha_error']
    beta_error = fit['beta_error']
    baseline_error = fit['baseline_error']

    relative_alpha = (alpha_error/alpha)
    relative_beta = (beta_error/beta)

    relative_error_lift = np.sqrt(relative_alpha**2 + relative_beta**2)
    lift30_error = lift30 * relative_error_lift

    lift = lift30/baseline
    lift_error = lift30_error/baseline

    print 'alpha: %0.4f +/- %0.4f' % (alpha, alpha_error)
    print 'beta: %s +/- %s days' % (beta, beta_error)
    print 'baseline: %0.4f +/- %0.4f' % (baseline, baseline_error)
    print 'Average daily lift over 30 days: %0.4f +/- %0.4f' % (lift30, lift30_error)
    print 'Lift: %0.4f +/- %0.4f' % (lift, lift_error)

    fitted_values = exponential_with_baseline(mean_days, baseline, alpha, beta)
    plt.plot(mean_days, fitted_values, color="magenta")
    plt.plot(mean_days, baseline+mean_days*0.0, color='gray', linestyle='--')
    title = title+' Lift: %0.3f +/- %0.3f' % (lift, lift_error)
    plt.title(title)