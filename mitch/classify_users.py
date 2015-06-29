from readers import read
import numpy as np
from collections import Counter


def user_average_spend_classifier(stats=None):
    if stats is None:
        stats = list(read('customer_stats'))
    avg_spends = []
    user_lookup = {}
    for stat in stats:
        avg_spend = stat['Ave Trx'].replace('$', '').replace(',', '').strip()
        user_id = stat['user_id']
        if not user_id:
            continue
        if avg_spend:
            spend = float(avg_spend)
            avg_spends.append(spend)
            user_lookup[user_id] = spend

    perc_33, perc_66 = np.percentile(avg_spends, [33, 66])
    #round to nearest $10
    perc_33 = 10 * int(perc_33/10.0)
    perc_66 = 10 * int(perc_66/10.0)

    print "33 percentile: %s, 66 percentile %s" % (perc_33, perc_66)

    def classifier(user_id):
        if not user_id:
            return None
        if user_id not in user_lookup:
            return None
        avg_spend = user_lookup[user_id]
        if avg_spend < perc_33:
            return 'low'
        if avg_spend < perc_66:
            return 'medium'
        return 'high'

    #add the meta data
    classifier.perc_33 = perc_33
    classifier.perc_66 = perc_66

    return classifier


def filter_messages(messages, user_class, user_classifier=None):
    if user_classifier is None:
        user_classifier = user_average_spend_classifier()

    return [m for m in messages if 'user_customer_external_id' in m
            and user_classifier(m['user_customer_external_id']) == user_class]


def count_messages_by_user_class(messages, user_classifier=None):
    if user_classifier is None:
        user_classifier = user_average_spend_classifier()

    count = Counter()
    for message in messages:
        if 'user_customer_external_id' not in message:
            continue
        user_id = message['user_customer_external_id']
        if not user_id:
            continue
        user_class = user_classifier(user_id)
        count[user_class] += 1
    for user_class, number in count.most_common():
        print "%s: %s" % (user_class, number)




