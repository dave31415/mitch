from readers import read
from itertools import groupby
from csv import DictWriter


def sales_grouped_by_users():
    keyfunc = lambda x: x['customer_external_id']
    sales = sorted(list(read('sales')), key=keyfunc)
    return groupby(sales, keyfunc)


def customer_stats(outfile=None):
    sales = sales_grouped_by_users()

    stats = {}
    for user_id, items in sales:
        item_list = list(items)
        data = {}
        data['n_lines'] = len(item_list)
        data['n_items'] = sum([int(i['quantity']) for i in item_list])
        #next two is count of unique
        data['n_orders'] = len({i['order_number'] for i in item_list})
        data['n_skus'] = len({i['sku'] for i in item_list})
        data['total_spend'] = sum([float(i['price']) for i in item_list])
        data['last_purchase'] = max(i['purchase_date'] for i in item_list)
        data['first_purchase'] = min(i['purchase_date'] for i in item_list)
        data['user_id'] = user_id
        stats[user_id] = data

    if outfile is not None:
        fieldnames = sorted(data.keys())
        dw = DictWriter(open(outfile, 'w'), fieldnames=fieldnames)
        dw.writeheader()
        for user_id, row in stats.iteritems():
            dw.writerow(row)

    return stats

if __name__ == '__main__':
    customer_stats(outfile='customer_stats.csv')
