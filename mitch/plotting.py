from matplotlib import pylab as plt


def plot_lift_data(lift_data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    alpha = [l['fit']['alpha'] for l in lift_data.values()]
    alpha_error = [l['fit']['alpha_error'] for l in lift_data.values()]
    beta = [l['fit']['beta'] for l in lift_data.values()]
    beta_error = [l['fit']['beta_error'] for l in lift_data.values()]
    message_class = lift_data.keys()

    ax.plot(beta, alpha, color='red', linestyle='', marker='o', markersize=10)
    ax.errorbar(beta, alpha, xerr=beta_error, yerr=alpha_error, linestyle='')

    for a, b, c in zip(alpha, beta, message_class):
        ax.annotate(c, xy=(b, a), xytext=(b+2, a+.01), fontsize=17)
    plt.xlim(0, max(beta)+30)
    plt.ylim(0, max(alpha)+.05)
    plt.xlabel('Duration (days)')
    plt.ylabel('Initial lift')
    plt.show()
