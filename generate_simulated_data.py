from constants_and_util import *
import itertools
import random
import pandas as pd
import geohash
import os
import datetime
import pytz
from timezonefinder import TimezoneFinder
from multiprocessing import Pool

def gen_lat_lon():
    lat = random.uniform(35, 45)
    lon = random.uniform(-120, -90)
    return lat, lon

def gen_safegraph_id(id_prefix):
    sfx = ['%02x' % random.randrange(256) for i in range(31)]
    return id_prefix + ''.join(sfx)

def gen_utc_timestamp(utc_day, utc_hour):
    utc_min = random.randrange(60)
    utc_sec = random.randrange(60)
    dt = datetime.datetime.fromisoformat(
        '%s %s:%02d:%02d+00:00' % (utc_day.replace('_', '-'), utc_hour, utc_min, utc_sec))
    return int(dt.strftime('%s'))

random.seed(987987234)
tzf = TimezoneFinder()

all_lat_lon = [gen_lat_lon() for i in range(50)]

def gen_home_loc():
    lat, lon = random.choice(all_lat_lon)
    lat += random.uniform(-0.0001, 0.0001)
    lon += random.uniform(-0.0001, 0.0001)
    return lat, lon

home_locations = {id_prefix: [gen_home_loc() for i in range(2)] for id_prefix in VALID_ID_PREFIXES}
all_home_locs = [ll for v in home_locations.values() for ll in v]

zillow_df = []

for i, (lat, lon) in enumerate(all_home_locs):
    zillow_df.append((
        lat,
        lon,
        lat,
        lon,
        random.uniform(100000, 1000000),
        random.uniform(500, 10000),
        random.uniform(10, 50),
        random.uniform(10, 50),
        'Alabama',
        'full_addr_%d' % i,
        'dummy',
        'dummy',
        'dummy',
        'dummy',
        'dummy',
    ))
zillow_df = pd.DataFrame(zillow_df, columns=[
    'inferred_home_location_lat', 'inferred_home_location_lon', 'zillow_lat', 'zillow_lon',
    'zestimate', 'rent_zestimate', 'zillow_distance', 'dist_cl',
    'state', 'full_addr', 'census_tract', 'city', 'addr_zip', 'addr_zip_first_5', 'use_code'])
zillow_df.to_csv(ZILLOW_QUERY_RESULTS_FILE, compression='gzip', index=False)

def gen_for_id_prefix(id_prefix):
    random.seed(id_prefix + "myseed")
    safegraph_ids = []
    lat_lons = []
    id_types = []
    timezones = []
    for i in range(2):
        safegraph_ids.append(gen_safegraph_id(id_prefix))
        lat, lon = home_locations[id_prefix][i]
        lat_lons.append((lat, lon))
        id_types.append('aaid' if random.randrange(2) else 'idfa')
        timezones.append(tzf.timezone_at(lat=lat, lng=lon))
    for utc_day, utc_hour in itertools.product(UTC_DAYS_TO_USE_IN_ANALYSIS, VALID_UTC_HOURS):
        fd, fn = generate_filepath('locations', id_prefix, utc_day, utc_hour)
        os.makedirs(fd, exist_ok=True)
        outfile_name = os.path.join(fd, fn)
        df = []
        for safegraph_id, (lat, lon), id_type, tz in zip(safegraph_ids, lat_lons, id_types, timezones):
            for j in range(3):
                utc_timestamp = gen_utc_timestamp(utc_day, utc_hour)
                local_tz = pytz.timezone(tz)
                utc_time = datetime_from_utc_timestamp(utc_timestamp)
                local_time = utc_time.astimezone(local_tz)
                local_time = local_tz.normalize(local_time).replace(tzinfo=None)
                df.append((
                    safegraph_id,
                    lat,
                    lon,
                    random.uniform(80, 105),
                    id_type,
                    utc_timestamp,
                    geohash.encode(lat, lon, 9),
                    utc_time.strftime(DATETIME_FORMATTER_STRING),
                    tz,
                    local_time.strftime(DATETIME_FORMATTER_STRING),
                ))
        df = pd.DataFrame(df, columns=['safegraph_id', 'latitude', 'longitude', 'horizontal_accuracy', 'id_type', 'utc_timestamp', 'geo_hash',
                                       'utc_datetime', 'timezone', 'local_datetime'])
        df.to_csv(outfile_name, compression='gzip', index=False)

with Pool(8) as pool:
    pool.map(gen_for_id_prefix, VALID_ID_PREFIXES)
