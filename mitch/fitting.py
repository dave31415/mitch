import scipy.optimize as opt
from lmfit import minimize, Parameters
import numpy as np


def exponential_with_baseline(x, baseline, amplitude, scale):
    return baseline + amplitude * np.exp(-x / scale)


def exponential_lift(alpha, beta, ndays=15.0):
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
    params.add('alpha', value=0.3, min=-0.3, max=4.0)
    params.add('beta', value=40.0, min=9.0, max=100.0)
    params.add('baseline', value=0.75, min=0.5, max=1.0)

    eps_data = ratio*0.1+0.1
    fit = minimize(exponential_residual, params, args=(days, ratio, eps_data))
    return fit


def fit_ratio(days, ratio, type='lmfit'):
    if type == 'scipy':
        return fit_ratio_scipy(days, ratio)
    if type == 'lmfit':
        fit = fit_ratio_lmfit(days, ratio)
        params = fit.params.valuesdict()
        pars = [params['baseline'], params['alpha'], params['beta']]
        cov = fit.covar
        return pars, cov
    raise ValueError('type must be scipy or lmfit, type=%s' % type)


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
        days_boot = days_boot[s]
        ratio_boot = ratio_boot[s]
        fit_boot = fit_ratio(days_boot, ratio_boot)
        alpha = fit_boot[0][1]
        beta = fit_boot[0][2]
        baseline = fit_boot[0][0]
        lift = exponential_lift(alpha, beta)
        data = {'alpha': alpha, 'beta': beta, 'baseline': baseline, 'lift': lift}
        boots.append(data)

    for param in result.keys():
        par = np.array([b[param] for b in boots])
        result[param+'_mean'] = par.mean()
        result[param+'_error'] = par.std()
    result['nboot'] = nboot
    return result, boots