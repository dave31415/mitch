from matplotlib import pylab as plt
import numpy as np


def plot_lift_data(lift_data):
    np.random.seed(42113)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    alpha = [l['fit']['alpha'] for l in lift_data.values()]
    alpha_error = [l['fit']['alpha_error'] for l in lift_data.values()]
    beta = [l['fit']['beta'] for l in lift_data.values()]
    beta_error = [l['fit']['beta_error'] for l in lift_data.values()]
    message_class = lift_data.keys()

    num = len(beta)
    beta_jitter = np.random.randn(num)
    np.random.seed(None)
    beta = np.array(beta) + beta_jitter

    ax.plot(beta, alpha, color='red', linestyle='', marker='o', markersize=10)
    ax.errorbar(beta, alpha, xerr=beta_error, yerr=alpha_error, linestyle='')

    for a, b, c in zip(alpha, beta, message_class):
        ax.annotate(c, xy=(b, a), xytext=(b+2, a+.01), fontsize=17)
    plt.xlim(0, max(beta)+30)
    plt.ylim(0, 0.9)
    plt.xlabel('Duration (days)')
    plt.ylabel('Initial Lift')
    plt.show()


def combine_keys(lift_data):
    combined_data = {}
    for message_class, dat1 in lift_data.iteritems():
        for user_class, data in dat1.iteritems():
            key = message_class+'_'+user_class
            combined_data[key] = data
    return combined_data