import numpy as np
from readers import read_merged_messages
from collections import defaultdict
import date_utils as du
from datetime import timedelta
from readers import read
from matplotlib import pylab as plt
import time
import scipy.optimize as opt
from lmfit import minimize, Parameters


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


def count_sales_days_since_message(messages=None, purchases=None,
                                   niter_random=1, doplot=True, color="blue"):
    buffer_days = 60

    start_time = time.time()

    if messages is None:
        print 'reading messages'
        messages = read_merged_messages()

    if purchases is None:
        print 'reading purchase dates'
        purchases = read('sales')

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
        if i % 1000 == 0:
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
    bins = binning_function(days, factor)
    data = {}

    for day, rat, bin in zip(days, ratio, bins):
        if bin not in data:
            data[bin] = {'num': 0, 'sum_days': 0.0, 'sum_ratio': 0.0}
        data[bin]['num'] += 1
        data[bin]['sum_days'] += day
        data[bin]['sum_ratio'] += rat
    for bin, vals in data.iteritems():
        vals['mean_days'] = vals['sum_days']/vals['num']
        vals['mean_ratio'] = vals['sum_ratio']/vals['num']
    return data


def exponential_with_baseline(x, baseline, amplitude, scale):
    return baseline + amplitude * np.exp(-x / scale)


def exponential_lift(alpha, beta, ndays=15):
    return alpha*np.exp(-ndays/beta)


def fit_ratio_scipy(days, ratio):
    func = exponential_with_baseline
    sigma = None
    guess = np.array([0.86, 0.32, 20.0])
    fit = opt.curve_fit(func, days, ratio, guess, sigma)
    return fit


def exponential_residual(params, x, data, eps_data):
    alpha = params['alpha'].value
    beta = params['beta'].value
    baseline = params['baseline'].value
    model = baseline + alpha * np.exp(-x/beta)
    return (data-model)/eps_data


def fit_ratio_lmfit(days, ratio):
    params = Parameters()
    params.add('alpha', value=0.25, min=-0.3, max=4.0)
    params.add('beta', value=40.0, min=9.0, max=70.0)
    params.add('baseline', value=0.85, min=0.5, max=1.3)

    eps_data = ratio*0.1+0.1
    fit = minimize(exponential_residual, params, args=(days, ratio, eps_data))
    return fit


def fit_ratio(days, ratio, type='scipy'):
    if type == 'scipy':
        return fit_ratio_scipy(days,ratio)
    else:
        pass


def boot_fit(days, ratio, nboot=100):
    boots = []
    fit = fit_ratio(days, ratio)
    alpha = fit[0][1]
    beta = fit[0][2]
    baseline = fit[0][0]
    lift = exponential_lift(alpha, beta)
    result = {'alpha': alpha, 'beta': beta, 'baseline': baseline, 'lift': lift}

    n_days = len(days)
    for boot in xrange(nboot):
        random_sample = np.random.randint(n_days, size=n_days)
        days_boot = days[random_sample]
        ratio_boot = ratio[random_sample]
        s = np.argsort(days_boot)
        days_boot =days_boot[s]
        ratio_boot = ratio_boot[s]
        plt.plot(days_boot, ratio_boot, alpha=0.2)
        fit_boot = fit_ratio(days_boot, ratio_boot)
        alpha = fit_boot[0][1]
        beta = fit_boot[0][2]
        baseline = fit_boot[0][0]
        lift = exponential_lift(alpha, beta)
        data = {'alpha': alpha, 'beta': beta, 'baseline': baseline, 'lift': lift}
        print data
        boots.append(data)

    for param in result.keys():
        par = np.array([b[param] for b in boots])
        result[param+'_mean'] = par.mean()
        result[param+'_sigma'] = par.std()
    result['nboot'] = nboot
    return result, boots


def plot_ratios(n_days_counter, n_days_counter_random, scale=1.0,
                color="gray", factor=9.0, nmax=200):
    days, ratio = get_days_ratio(n_days_counter, n_days_counter_random,
                                 scale=scale, nmax=nmax)

    plt.plot(days, ratio, color=color, alpha=0.3)
    plt.xlabel('N days since message')
    plt.ylabel('Sales Lift')

    data = bin_days_ratio(days, ratio, factor=factor)
    mean_days = np.array([i['mean_days'] for i in data.values()])
    mean_ratio = np.array([i['mean_ratio'] for i in data.values()])
    plt.plot(mean_days, mean_ratio, color="blue", marker='o', markersize=8)

    fit = fit_ratio(days, ratio)
    params = fit[0]
    baseline = params[0]
    alpha = params[1]
    beta = params[2]

    lift30 = exponential_lift(alpha, beta)
    alpha_error = np.sqrt(fit[1][1, 1])
    beta_error = np.sqrt(fit[1][2, 2])

    relative_alpha = (alpha_error/alpha)
    relative_beta = (beta_error/beta)

    relative_error_lift = np.sqrt(relative_alpha**2 + relative_beta**2)
    lift30_error = lift30 * relative_error_lift

    print 'alpha: %0.4f +/- %0.4f' % (alpha, alpha_error)
    print 'beta: %s days, baseline: %s' % (beta, baseline)
    print 'Average daily lift over 30 days: %0.4f +/- %0.4f' % (lift30, lift30_error)

    fitted_values = exponential_with_baseline(days, *params)
    plt.plot(days, fitted_values, color="magenta")
    plt.plot(days, baseline+days*0.0, color='gray', linestyle='--')


    print fit
