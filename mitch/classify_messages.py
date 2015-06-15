def classify_messages(message_name):
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


def filter_messages(messages, message_class):
    return [m for m in messages if 'campaign_name' in m
            and classify_messages(m['campaign_name']) == message_class]

