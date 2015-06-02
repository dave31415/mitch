from params import root_dir
from csv import DictReader


def file_names():
    files = {
        'sales': "%s/sales_20150528.csv" % root_dir,
        'users': "%s/spree_users_20150528.csv" % root_dir,
        'campaigns': "%s/product_list.csv" % root_dir,
        'messages': "%s/product_lists_users_20150528.csv" % root_dir,
    }
    return files


def read(name):
    files = file_names()
    return DictReader(open(files[name], 'r'))
