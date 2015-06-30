import readers
import geopy

from pyzipcode import ZipCodeDatabase
#read database once
zcdb = ZipCodeDatabase()


def zip_for_store(store_num):
    store_num = int(store_num)

    lookup = {1: '06880',
              2: '06830',
              3: '11743',
              11: '94108',
              12: '94304'}

    if store_num not in lookup:
        return None
    return lookup[store_num]


def make_customer_external_map():
    #returns a function which takes customer_external_id and
    # maps it to customer_id
    # or None if not found in household data
    households = readers.read('households')
    lookup = {}

    for house in households:
        lookup[house['external_id']] = house['id']

    def func(customer_external_id):
        if customer_external_id in lookup:
            return lookup[customer_external_id]
        else:
            print "can't find customer_external_id: %s in households" % customer_external_id
            return None

    return func


def make_zipcode_map():
    customer_map = make_customer_external_map()
    addresses = readers.read('addresses')
    lookup = {}
    for address in addresses:
        lookup[address['id']] = address['zipcode']

    def zfunc(customer_external_id):
        customer_id = customer_map(customer_external_id)
        if customer_id is None:
            return None
        if customer_id in lookup:
            return lookup[customer_id]
        else:
            print "can't find customer_id: %s in addresses" % customer_id
            return None
    return zfunc


def distance_of_purchase(customer_external_id, store_number, zipcode_map):
    zip_address = zipcode_map(customer_external_id)
    if zip_address is None:
        return None
    zip_store = zip_for_store(store_number)
    if zip_store is None:
        return None

    loc_address = zcdb.get(zip_address)
    if loc_address is None:
        return None
    loc_address = loc_address[0]

    loc_store = zcdb.get(zip_store)
    if loc_store is None:
        return None
    loc_store = loc_store[0]

    address_lat_long = (loc_address.latitude, loc_address.longitude)
    store_lat_long = (loc_store.latitude, loc_store.longitude)

    distance_miles = geopy.distance.vincenty(address_lat_long, store_lat_long).miles

    return distance_miles


