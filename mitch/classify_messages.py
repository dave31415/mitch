from collections import Counter


def classify_messages_old(message_name):
    #deprecated
    message_name = message_name.lower()
    if 'women' in message_name:
        return 'women'
    if 'men' in message_name:
        return 'men'
    if 'calendar' in message_name:
        return 'calendar'
    if 'sale' in message_name:
        return 'sale'
    if 'anniversary' in message_name:
        return 'anniversary'
    if 'birth' in message_name:
        return 'birthday'
    if 'day' in message_name:
        return 'holiday'

    return 'other'


def classify_messages(message_name):
    message_name = message_name.lower()

    events_tags = ['preview', 'calendar', 'event', 'trunk show',
                   'save-the-date', 'luncheon']
    for et in events_tags:
        if et in message_name:
            return 'event'
    if 'sale' in message_name:
        return 'sale'
    if 'birth' in message_name:
        return 'birthday'
    if 'anniversary' in message_name:
        return 'anniversary'
    if 'holiday' in message_name:
        return 'holiday'

    return 'other'


def filter_messages(messages, message_class):
    return [m for m in messages if 'campaign_name' in m
            and classify_messages(m['campaign_name']) == message_class]


def count_messages_by_message_class(messages):
    count = Counter()
    for message in messages:
        if 'campaign_name' not in message:
            continue
        message_name = message['campaign_name']
        if not message_name:
            continue
        message_class = classify_messages(message_name)
        count[message_class] += 1
    for message_class, number in count.most_common():
        print "%s: %s" % (message_class, number)
