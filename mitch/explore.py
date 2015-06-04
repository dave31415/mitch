from readers import read, read_merged_messages
from datetime import datetime
from collections import defaultdict

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
        user_id = message['user_id']
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


def sales_days_from_messages():
    date_format = "%m/%d/%Y %H:%M:%S"
    messages = read_merged_messages()
    n_messages = len(messages)
    purchase_dates = get_purchase_dates()
    days_between_purchase_and_message = []

    for i, message in enumerate(messages):
        if (i % 1000) == 0:
            print "%s of %s" % (i, n_messages)
        user_id = message['user_id']
        if user_id in purchase_dates:
            user_purchase_dates = purchase_dates[user_id]

            if 'campaign_sent_at' in message:
                date_string = message['campaign_sent_at']
                message_date = datetime.strptime(date_string, date_format)
                for user_purchase_date in user_purchase_dates:
                    diff_days = user_purchase_date - message_date
                    days_between_purchase_and_message.append(diff_days)

    return days_between_purchase_and_message



