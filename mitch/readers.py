from params import data_dir
from csv import DictReader


def file_names():
    files = {
        'sales': "%s/sales_20150528.csv" % data_dir,
        'users': "%s/spree_users_20150528.csv" % data_dir,
        'campaigns': "%s/product_list.csv" % data_dir,
        'messages': "%s/product_lists_users_20150528.csv" % data_dir,
    }
    return files


def read(name):
    files = file_names()
    return DictReader(open(files[name], 'rU'))


def read_merged_messages():
    """
    reads the messages file and joins it to users and campaigns
    only keeps a subset of columns
    """

    messages = list(read('messages'))
    campaigns = read('campaigns')
    users = read('users')

    user_fields = ['id', 'firstname', 'lastname', 'household_id', 'client']
    campaign_fields = ['id', 'name', 'sent_at', 'seller_id', 'store_id',
                       'created_at', 'updated_at', 'type', 'parent_id']

    user_lookup = {line['id']: {'user_'+i: line[i] for i in user_fields} for line in users}
    campaign_lookup = {line['id']: {'campaign_'+i: line[i] for i in campaign_fields} for line in campaigns}

    missing_users = 0
    missing_campaigns = 0

    for mess in messages:
        user_id = mess['user_id']
        if user_id in user_lookup:
            mess.update(user_lookup[user_id])
        else:
            print "%s not in user_lookup" % user_id
            missing_users += 1

        campaign_id = mess['product_list_id']

        if campaign_id in campaign_lookup:
            mess.update(campaign_lookup[campaign_id])
        else:
            print "%s not in campaign_lookup" % campaign_id
            missing_campaigns += 1

    print "%s missing users" % missing_users
    print "%s missing campaigns" % missing_campaigns

    return messages
