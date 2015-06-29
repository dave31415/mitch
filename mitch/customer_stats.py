from readers import read
from itertools import groupby
from csv import DictWriter
import numpy as np
from matplotlib import pylab as plt


def sales_grouped_by_users():
    keyfunc = lambda x: x['customer_external_id']
    sales = sorted(list(read('sales')), key=keyfunc)
    return groupby(sales, keyfunc)


def fill_items(data, item_list, suffix=''):
    if len(item_list) == 0:
        data['n_items'+suffix] = 0
        data['n_orders'+suffix] = 0
        data['n_skus'+suffix] = 0
        data['total_spend'+suffix] = 0.0
        data['last_purchase'+suffix] = None
        data['first_purchase'+suffix] = None
        data['avg_spend'+suffix] = None
    else:
        data['n_items'+suffix] = sum([int(i['quantity']) for i in item_list])
        data['n_orders'+suffix] = len({i['order_number'] for i in item_list})
        data['n_skus'+suffix] = len({i['sku'] for i in item_list})
        data['total_spend'+suffix] = sum([float(i['price']) for i in item_list])
        data['last_purchase'+suffix] = max(i['purchase_date'] for i in item_list)
        data['first_purchase'+suffix] = min(i['purchase_date'] for i in item_list)
        if data['n_items'+suffix] == 0:
            data['avg_spend'+suffix] = None
        else:
            data['avg_spend'+suffix] = data['total_spend'+suffix]/data['n_items'+suffix]


def customer_stats(outfile=None):
    sales = sales_grouped_by_users()

    stats = {}
    for user_id, items in sales:
        item_list = list(items)
        data = {}
        data['user_id'] = user_id
        data['n_lines'] = len(item_list)
        #all orders
        fill_items(data, item_list, suffix='')
        #online orders
        item_list_online = [i for i in item_list if i['online_order_number']]
        fill_items(data, item_list_online, suffix='_online')
        # sale items
        item_list_on_sale = [i for i in item_list if i['on_sale'] == 't']
        fill_items(data, item_list_on_sale, suffix='_on_sale')

        stats[user_id] = data

    if outfile is not None:
        fieldnames = sorted(data.keys())
        dw = DictWriter(open(outfile, 'w'), fieldnames=fieldnames)
        dw.writeheader()
        for user_id, row in stats.iteritems():
            dw.writerow(row)

    return stats.values()


def my_hist(data, bins, range, field, color, label):
    #plt.hist(data, bins, label=label, alpha=0.2, normed=True,
    #         range=range, color=color)
    y, bin_edges = np.histogram(data, bins=bins, normed=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    #plt.plot(bin_centers, y, '--', color=color, alpha=0.5)
    plt.fill_between(bin_centers, y, color=color, alpha=0.2, label=label)
    plt.xlabel('$log_{10}($ ' + field+' $)$')
    plt.ylabel('Density')


def stats_by_online_or_not(stats):
    stats_online = [s for s in stats if s['first_purchase_online']]
    stats_not_online = [s for s in stats if s['first_purchase_online'] is None]

    compare_two_groups(stats_online, stats_not_online,
                       'Online', 'Not online', bins=25)


def stats_by_online_first_or_not(stats):
    stats_online_first = [s for s in stats if s['first_purchase_online']
                          and s['first_purchase_online'] == s['first_purchase']]

    stats_not_online_first = [s for s in stats if s['first_purchase_online']
                          and s['first_purchase_online'] != s['first_purchase']]

    print 'N online first: %s' % len(stats_online_first)
    print 'N in store first: %s' % len(stats_not_online_first)

    compare_two_groups(stats_online_first, stats_not_online_first,
                       'Online first', 'Online after store', bins=15)


def compare_two_groups(stats_1, stats_2, label_1, label_2, bins=25):

    fields = ['n_orders', 'n_skus', 'total_spend', 'avg_spend']
    plt.clf()
    i = 0

    for field in fields:
        var_1 = np.array([s[field] for s in stats_1 if s[field] is not None])
        var_2 = np.array([s[field] for s in stats_2 if s[field] is not None])

        var_1_log10 = np.log10(1+abs(var_1))
        var_2_log10 = np.log10(1+abs(var_2))
        i += 1
        ax = plt.subplot(2, 2, i)
        range = [var_1_log10.min(), var_2_log10.max()]

        my_hist(var_1_log10, bins, range, field, 'blue', label_1)
        my_hist(var_2_log10, bins, range, field, 'green', label_2)

        if i == 1:
            #hack to get legend to work with fill_between rather than hist
            #be sure they are in the right order as the plots!!
            p1 = plt.Rectangle((0, 0), 1, 1, fc='blue', alpha=0.2)
            p2 = plt.Rectangle((0, 0), 1, 1, fc='green', alpha=0.2)
            plt.legend([p1, p2], [label_1, label_2])

    plt.show()


if __name__ == '__main__':
    customer_stats(outfile='customer_stats.csv')
