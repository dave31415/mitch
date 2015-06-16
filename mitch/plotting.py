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


def bar_charts(lift_data, var='alpha'):
    plt.clf()
    bar_width = 0.5
    message_classes = [m.capitalize() for m in lift_data.keys()]
    x = [d['fit'][var] for d in lift_data.values()]
    index = np.arange(len(message_classes))

    plt.bar(index + bar_width, x, bar_width, alpha=0.5)
    plt.xticks(index + 1.5*bar_width, message_classes)
    plt.xlim(0, max(index + bar_width))
    if var == 'alpha':
        plt.ylabel("Initial Lift")
    else:
        plt.ylabel("Duration (days)")


def bar_charts_with_range(lift_data, var='alpha'):
    plt.clf()
    bar_width = 0.5
    message_classes = [m.capitalize() for m in lift_data.keys()]
    x = np.array([d['fit'][var] for d in lift_data.values()])
    xerr = np.array([d['fit'][var+'_error'] for d in lift_data.values()])
    xmin = x - xerr
    #xmax = x + xerr
    height = xerr*2
    index = np.arange(len(message_classes)) + bar_width

    plt.bar(index, height, bar_width, xmin, alpha=0.5)
    plt.xticks(index + 0.5*bar_width, message_classes)
    plt.xlim(0, max(index))
    plt.ylim(ymin=0)
    plt.plot(index + 0.5*bar_width, x, 'ro')
    plt.plot(index + 0.5*bar_width, x, color='red', marker='_',
             linestyle='', markersize=22)
    if var == 'alpha':
        plt.ylabel("Initial Lift")
    else:
        plt.ylabel("Duration (days)")



