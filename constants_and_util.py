import os
import fcntl
from multiprocessing.pool import ThreadPool
import dask

MAX_NUMPY_CORES = 8
print("Setting numpy cores to %i" % MAX_NUMPY_CORES)
os.environ["MKL_NUM_THREADS"] = str(MAX_NUMPY_CORES) # this keeps numpy from using every available core. We have to do this BEFORE WE import numpy for the first time. 
os.environ["NUMEXPR_NUM_THREADS"] = str(MAX_NUMPY_CORES)
os.environ["OMP_NUM_THREADS"] = str(MAX_NUMPY_CORES)
dask.config.set(pool=ThreadPool(MAX_NUMPY_CORES)) # This is to make Dask play nicely with the thread limit. See: 


import datetime
import numpy as np
import sys
from itertools import product
from scipy.interpolate import interp1d
from datetime import timezone
import pytz
import collections
import time
import errno
import pandas as pd
import logging
import random
from scipy.stats import rankdata, spearmanr
import pandas as pd
import platform
import resource
import dask.dataframe as dd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from dask.diagnostics import ProgressBar
import tqdm

assert sys.version.split()[0] == '3.7.2'

np.random.seed(0)
random.seed(0)

MONTHS_TO_USE_IN_ANALYSIS = [3, 7, 11]
DATETIME_FORMATTER_STRING = '%Y-%m-%d %H:%M:%S'

BASE_DIRECTORY_FOR_EVERYTHING = 'base_dir'
BASE_DATA_DIR = os.path.join(BASE_DIRECTORY_FOR_EVERYTHING, 'newdata/')
USER_SPECIFIC_FEATURES_DIR = os.path.join(BASE_DATA_DIR, 'reorganized_data/user_specific_features/') # information about each user. 
MIDDLE_OF_NIGHT_LOCATIONS_DIR = os.path.join(USER_SPECIFIC_FEATURES_DIR, 'middle_of_night_locations')  # inferred home lcoations. 
HOME_LOCATIONS_DIR = os.path.join(USER_SPECIFIC_FEATURES_DIR, 'home_locations')
ZILLOW_QUERY_RESULTS_FILE = os.path.join(BASE_DIRECTORY_FOR_EVERYTHING, 'zillow_query_results.csv.gz')

PATH_CROSSINGS_DIR = os.path.join(BASE_DATA_DIR, 'reorganized_data/path_crossings/') # pairs of users whose paths cross. 
ANNOTATED_PATH_CROSSINGS_DIR = os.path.join(BASE_DATA_DIR, 'reorganized_data/annotated_path_crossings/') # pairs of users whose paths cross, annotated with features like whether they're on a road. 
STRATIFIED_PATH_CROSSINGS_DIR = os.path.join(BASE_DATA_DIR, 'reorganized_data/stratified_path_crossings/') # break down the path crossings dataset into subsets to make it faster to parallelize. 
FRACTION_OF_DUPLICATE_PINGS_PER_USER_PATH = os.path.join(BASE_DATA_DIR, 'reorganized_data/path_crossings/fraction_of_duplicate_pings_per_user.csv.gz')
PATH_CROSSING_CONSOLIDATION_INTERVAL = 300 # (in seconds) Consolidate all path crossings in each 5 minute interval into a single one.
PATH_CROSSING_STATIONARY_THRESHOLDS = [1, 2, 3, 4, 5, 10, 50] # (in meters) Thresholds between locations to consider as stationary.
MAX_PATH_CROSSING_THRESHOLD_TO_USE_IN_INITIAL_FILTERING = 50 # when we initially compute path crossings, define a user pair as crossing paths if they come within this many meters of each other. 
DUPLICATE_PINGS_BLACKLIST_FRACTION_THRESHOLD = 0.8 # Blacklist users where over 80% of their pings are duplicate with another user (identical lat, lon, timestamp).
PathCrossingThreshold = collections.namedtuple('PathCrossingThreshold', ['distance_meters', 'time_seconds'])
PATH_CROSSING_ADDITIONAL_THRESHOLDS = [
    PathCrossingThreshold(distance_meters=50, time_seconds=300),
    PathCrossingThreshold(distance_meters=50, time_seconds=120),
    PathCrossingThreshold(distance_meters=50, time_seconds=60),
    PathCrossingThreshold(distance_meters=25, time_seconds=300),
    PathCrossingThreshold(distance_meters=25, time_seconds=120),
    PathCrossingThreshold(distance_meters=10, time_seconds=300),
    PathCrossingThreshold(distance_meters=10, time_seconds=60),
]
SES_MEASURES_FOR_ECONOMIC_SEGREGATION = ['rent_zestimate']

FeatureType = collections.namedtuple('FeatureType', ['feature_type', 'display_name', 'noun'])

LEISURE_FEATURES = [
    FeatureType(feature_type='safegraph_places_feature_full_service_restaurants', display_name='Full-Service Restaurants', noun='restaurant'),
    FeatureType(feature_type='safegraph_places_feature_snack_and_nonalcoholic_beverage_bars', display_name='Snack Bars', noun='snack bar'),
    FeatureType(feature_type='safegraph_places_feature_limited_service_restaurants', display_name='Limited-Service Restaurants', noun='restaurant'),
    FeatureType(feature_type='safegraph_places_feature_sports_teams_and_clubs', display_name='Stadiums', noun='stadium'),
    FeatureType(feature_type='safegraph_places_feature_promoters_of_performing_arts_sports_and_similar_events_with_facilities', display_name='Performing Arts Centers', noun='center'),
    FeatureType(feature_type='safegraph_places_feature_fitness_and_recreational_sports_centers', display_name='Fitness/Recreation Centers', noun=''),
    FeatureType(feature_type='safegraph_places_feature_historical_sites', display_name='Historical Sites', noun=''),
    FeatureType(feature_type='safegraph_places_feature_amusement_and_theme_parks', display_name='Theme Parks', noun=''),
    FeatureType(feature_type='safegraph_places_feature_drinking_places_alcoholic_beverages', display_name='Bars/Drinking Places', noun=''),
    FeatureType(feature_type='safegraph_places_feature_nature_parks_and_other_similar_institutions', display_name='Parks', noun=''),
    FeatureType(feature_type='safegraph_places_feature_religious_organizations', display_name='Religious Organizations', noun=''),
    FeatureType(feature_type='safegraph_places_feature_bowling_centers', display_name='Bowling Centers', noun=''),
    FeatureType(feature_type='safegraph_places_feature_museums', display_name='Museums', noun=''),
    FeatureType(feature_type='safegraph_places_feature_casinos_except_casino_hotels', display_name='Casinos', noun=''),
    FeatureType(feature_type='safegraph_places_feature_independent_artists_writers_and_performers', display_name='Independent Artists', noun=''),
    FeatureType(feature_type='safegraph_places_feature_all_other_amusement_and_recreation_industries', display_name='Other Amusement/Recreation', noun=''),
    FeatureType(feature_type='safegraph_places_feature_golf_courses_and_country_clubs', display_name='Golf Courses and Country Clubs', noun=''),



]

COMBINED_FEATURES = [
    ('other', [f.feature_type for f in LEISURE_FEATURES]),
]
TEMPORAL_PATH_CROSSING_THRESHOLDS = [
    ('num_consecutive_in_group', 3),  # 15 minutes
    ('num_consecutive_in_group', 6),  # 30 minutes
    ('num_consecutive_in_group', 9),
    ('num_consecutive_in_group', 12), # 60 minutes
    ('num_consecutive_in_group', 15),
    ('num_consecutive_in_group', 18),
    ('num_consecutive_in_group', 21),
    ('num_consecutive_in_group', 24), # 120 minutes
    ('num_consecutive_in_group', 36), # 180 minutes
    ('num_groups', 3),                # 15 minutes
    ('num_groups', 6),                # 30 minutes
    ('num_groups', 9),
    ('num_groups', 12),               # 60 minutes
    ('num_groups', 15),
    ('num_groups', 18),
    ('num_groups', 21),
    ('num_groups', 24),               # 120 minutes
    ('num_groups', 36),               # 180 minutes
    ('num_unique_days', 2),
    ('num_unique_days', 3),
    ('num_unique_days', 4),
    ('num_unique_days', 5),
    ('num_unique_days', 6),
    ('num_unique_days', 7),
    ('num_unique_days', 8),
    ('num_unique_days', 9),
    ('num_unique_days', 10),
]
ECONOMIC_SEGREGATION_MEASURES_DIR = os.path.join(BASE_DIRECTORY_FOR_EVERYTHING, 'outfiles_and_results/economic_segregation_measures/')
SAFEGRAPH_PLACES_DIR = os.path.join(BASE_DIRECTORY_FOR_EVERYTHING, '20191213-safegraph-aggregate-longitudinal-data-to-unzip-to/SearchofAllRecords-CORE_POI-GEOMETRY-PATTERNS-2017_07-2019-12-12')
SAFEGRAPH_PLACES_NAICS6_MAPPING_FILE = os.path.join(BASE_DIRECTORY_FOR_EVERYTHING, 'naics6_safegraph_places.csv')
SAFEGRAPH_PLACES_NAICS_PREFIXES = ['7', '813', '531120'] # the prefixes we care about

HORIZONTAL_ACCURACY_THRESHOLD = 100 # filter out pings less accurate than this throughout the analysis. 
WGS_84_CRS = {'init' :'epsg:4326'} 
MIDDLE_OF_NIGHT_HYPERPARAMS = {'local_start_hour':18, 
                               'local_end_hour':9, 
                               'weekdays_only':False, 
                               'min_nights_required':3, 
                               'min_total_pings_a_user_must_have':500, 
                               'max_distance':50, 
                               'min_frac_near_median':0.6}
ALL_FEATURE_MAPPING_DATA_SOURCES = ['safegraph_places', 'tiger_rails', 'tiger_roads', 'home_census']

CRS_FOR_DISTANCE_COMPUTATIONS_IN_GEOPANDAS_IN_NORTH_AMERICA = "+proj=eqdc +lat_0=39 +lon_0=-96 +lat_1=33 +lat_2=45 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"

TIGER_FEATURE_CODES = {
    "R1011": "railroad_feature",  # "Railroad Feature (Main, Spur, or Yard)",
    "R1051": "carline__streetcar_track__monorail__other_mass_transit_rail",  # "Carline, Streetcar Track, Monorail, Other Mass Transit Rail",
    "R1052": "cog_rail_line__incline_rail_line__tram",  # "Cog Rail Line, Incline Rail Line, Tram",
    "S1100": "primary_road",  # "Primary Road",
    "S1200": "secondary_road",  # "Secondary Road",
    "S1400": "local_neighborhood_road__rural_road__city_street",  # "Local Neighborhood Road, Rural Road, City Street",
    "S1500": "vehicular_trail_4WD",  # "Vehicular Trail (4WD)",
    "S1630": "ramp",  # "Ramp",
    "S1640": "service_drive_usually_along_a_limited_access_highway",  # "Service Drive usually along a limited access highway",
    "S1710": "walkway__pedestrian_trail",  # "Walkway/Pedestrian Trail",
    "S1720": "stairway",  # "Stairway",
    "S1730": "alley",  # "Alley",
    "S1740": "private_road_for_service_vehicles",  # "Private Road for service vehicles (logging, oil fields, ranches, etc.)",
    "S1750": "internal_US_census_bureau_use",  # "Internal U.S. Census Bureau use",
    "S1780": "parking_lot_road",  # "Parking Lot Road",
    "S1820": "bike_path_or_trail",  # "Bike Path or Trail",
    "S1830": "bridle_path",  # "Bridle Path",
    "S2000": "road_median"  # "Road Median"
}

FIPS_CODES_FOR_50_STATES_PLUS_DC = { # https://gist.github.com/wavded/1250983/bf7c1c08f7b1596ca10822baeb8049d7350b0a4b
    "10": "Delaware",
    "11": "Washington, D.C.",
    "12": "Florida",
    "13": "Georgia",
    "15": "Hawaii",
    "16": "Idaho",
    "17": "Illinois",
    "18": "Indiana",
    "19": "Iowa",
    "20": "Kansas",
    "21": "Kentucky",
    "22": "Louisiana",
    "23": "Maine",
    "24": "Maryland",
    "25": "Massachusetts",
    "26": "Michigan",
    "27": "Minnesota",
    "28": "Mississippi",
    "29": "Missouri",
    "30": "Montana",
    "31": "Nebraska",
    "32": "Nevada",
    "33": "New Hampshire",
    "34": "New Jersey",
    "35": "New Mexico",
    "36": "New York",
    "37": "North Carolina",
    "38": "North Dakota",
    "39": "Ohio",
    "40": "Oklahoma",
    "41": "Oregon",
    "42": "Pennsylvania",
    "44": "Rhode Island",
    "45": "South Carolina",
    "46": "South Dakota",
    "47": "Tennessee",
    "48": "Texas",
    "49": "Utah",
    "50": "Vermont",
    "51": "Virginia",
    "53": "Washington",
    "54": "West Virginia",
    "55": "Wisconsin",
    "56": "Wyoming",
    "01": "Alabama",
    "02": "Alaska",
    "04": "Arizona",
    "05": "Arkansas",
    "06": "California",
    "08": "Colorado",
    "09": "Connecticut",
    }
JUST_50_STATES_PLUS_DC = {'Alabama',
                         'Alaska',
                         'Arizona',
                         'Arkansas',
                         'California',
                         'Colorado',
                         'Connecticut',
                         'Delaware',
                         'Florida',
                         'Georgia',
                         'Hawaii',
                         'Idaho',
                         'Illinois',
                         'Indiana',
                         'Iowa',
                         'Kansas',
                         'Kentucky',
                         'Louisiana',
                         'Maine',
                         'Maryland',
                         'Massachusetts',
                         'Michigan',
                         'Minnesota',
                         'Mississippi',
                         'Missouri',
                         'Montana',
                         'Nebraska',
                         'Nevada',
                         'New Hampshire',
                         'New Jersey',
                         'New Mexico',
                         'New York',
                         'North Carolina',
                         'North Dakota',
                         'Ohio',
                         'Oklahoma',
                         'Oregon',
                         'Pennsylvania',
                         'Rhode Island',
                         'South Carolina',
                         'South Dakota',
                         'Tennessee',
                         'Texas',
                         'Utah',
                         'Vermont',
                         'Virginia',
                         'Washington',
                         'Washington, D.C.',
                         'West Virginia',
                         'Wisconsin',
                         'Wyoming'}

def get_valid_id_prefixes():
    """
    Each ID prefix is comprised of two hexadecimal digits; thus, there are 256 ID prefixes total. 
    Assuming each safegraph id is randomly generated, each prefix represents a random 256th of the dataset. 
    """
    VALID_ID_PREFIXES = list(map(''.join, product('0123456789abcdef', repeat=2)))
    assert len(VALID_ID_PREFIXES) == 256
    return VALID_ID_PREFIXES

def get_valid_utc_days():
    """
    List of valid UTC days as strings in the format YYYY_MM_DD. Eg, "2017_03_01". 
    """
    VALID_UTC_DAYS = []
    current_date = datetime.datetime(2017, 1, 1)
    while current_date.year == 2017:
        VALID_UTC_DAYS.append(current_date.strftime('%Y_%m_%d'))
        current_date += datetime.timedelta(days=1)
    assert len(VALID_UTC_DAYS) == 365
    return VALID_UTC_DAYS

def get_valid_utc_hours():
    """
    List of valid UTC hours as strings in the format DD. Eg, "00" and "17". 
    """
    VALID_UTC_HOURS = ["%02d" % d for d in range(24)]
    return VALID_UTC_HOURS

def get_analysis_utc_days():
    ANALYSIS_UTC_DAYS = []
    current_date = datetime.datetime(2017, 1, 1)
    while current_date.year == 2017:
        if current_date.month in MONTHS_TO_USE_IN_ANALYSIS:
            ANALYSIS_UTC_DAYS.append(current_date.strftime('%Y_%m_%d'))
        current_date += datetime.timedelta(days=1)
    return ANALYSIS_UTC_DAYS

VALID_UTC_DAYS = get_valid_utc_days()
VALID_UTC_HOURS = get_valid_utc_hours()
VALID_ID_PREFIXES = get_valid_id_prefixes()
UTC_DAYS_TO_USE_IN_ANALYSIS = get_analysis_utc_days()

def datetime_from_utc_timestamp(timestamp):
    """
    Return the UTC datetime corresponding to the given POSIX timestamp.
    """
    return datetime.datetime.utcfromtimestamp(timestamp)

def dtype_pandas_series(obj):
    return str(type(obj)) == "<class 'pandas.core.series.Series'>"

def lonlat_to_xyz(lon, lat):
    """
    Convert longitude and latitude coordinates to XYZ space.
    The XYZ space is preferred for computing distances using the Euclidean distance metric.
    (eg, sqrt((x1 - x2) ** 2 +  (y1 - y2) ** 2 +  (z1 - z2) ** 2).
    
    Computing distances using latitudes and longitudes is tricky. 
    See discussion here: https://stackoverflow.com/a/1185413/9477154
    The straight-line distance has a high margin of error for points far away on the earth's surface,
    since the measure of distance would go through the earth rather than across its surface.
    This conversion is fine for points that are close by because the earth is locally flat. 
    lon and lat can be either floats or arrays. 

    Confirmed that the distance we get for two test points with this method is very close to http://www.5thandpenn.com/GeoMaps/GMapsExamples/distanceComplete2.html
    which uses a more sophisticated method. (ie, within 0.1%). 
    """
    assert not dtype_pandas_series(lon)
    assert not dtype_pandas_series(lat)

    lon = lon * np.pi / 180 # convert to radians. 
    lat = lat * np.pi / 180
    RADIUS_EARTH = 6371 * 1e3 # radius in meters, so distances are in meters. https://en.wikipedia.org/wiki/Earth_radius
    x = RADIUS_EARTH * np.cos(lat) * np.cos(lon)
    y = RADIUS_EARTH * np.cos(lat) * np.sin(lon)
    z = RADIUS_EARTH * np.sin(lat)
    return x, y, z

def compute_distance_between_two_lat_lons(lat1, lat2, lon1, lon2):
    """
    converts to XYZ and computes straight line distance. Should work with either arrays or floats. 
    """
    assert not dtype_pandas_series(lat1)
    assert not dtype_pandas_series(lat2)
    assert not dtype_pandas_series(lon1)
    assert not dtype_pandas_series(lon2)
    
    x1, y1, z1 = lonlat_to_xyz(lat=lat1, lon=lon1)
    x2, y2, z2 = lonlat_to_xyz(lat=lat2, lon=lon2)
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

def generate_filepath(locations_or_visits, id_prefix, utc_day, utc_hour):
    """
    Gives us the path for a locations or visits file
    for a given id_prefix, utc_day, and utc_hour.
    """
    assert locations_or_visits in ['locations', 'visits']
    assert id_prefix in VALID_ID_PREFIXES
    assert utc_day in VALID_UTC_DAYS
    assert utc_hour in VALID_UTC_HOURS
    
    file_directory = os.path.join(BASE_DATA_DIR,
                                  'reorganized_data',
                                  locations_or_visits,
                                  'id_prefix_%s' % id_prefix,
                                  utc_day)

    file_name = '%s_id_prefix_%s_utc_day_%s_utc_hour_%s.csv.gz' % (
        locations_or_visits, id_prefix, utc_day, utc_hour)

    return file_directory, file_name

def load_ping_file(locations_or_visits=None, id_prefix=None, utc_day=None, utc_hour=None, file_path=None, usecols=None, use_dask=False, filter_horizontal_acc=True):
    """
    Preferred way to load ping file(s) because it applies filtering eg for horizontal accuracy. 

    Can be called either using locations_or_visits, id_prefix, utc_day, and utc_hour
    or 
    file_path (either a string or list of multiple paths) 
    """ 
    if file_path is None:
        file_directory, file_name = generate_filepath(locations_or_visits, id_prefix, utc_day, utc_hour)
        file_path = os.path.join(file_directory, file_name)
    else:
        assert locations_or_visits is None
        assert id_prefix is None
        assert utc_day is None
        assert utc_hour is None
    if isinstance(file_path, list):
        d = load_csv_possibly_with_dask(file_path, use_dask=use_dask, usecols=usecols)
    else:
        d = pd.read_csv(file_path, compression='gzip', usecols=usecols)
    if filter_horizontal_acc:
        d = d.loc[d['horizontal_accuracy'] <= HORIZONTAL_ACCURACY_THRESHOLD]
    d.index = range(len(d))
    return d

def load_csv_possibly_with_dask(filenames, use_dask=False, compression='gzip', blocksize=None, compute_with_dask=True, **kwargs):
    if not ('usecols' in kwargs and kwargs['usecols'] is not None):
        kwargs['usecols'] = lambda col: col != 'Unnamed: 0'
    if use_dask:
        with ProgressBar():
            d = dd.read_csv(filenames, compression=compression, blocksize=blocksize, **kwargs)
            if compute_with_dask:
                d = d.compute()
                d.index = range(len(d))
            return d
    else:
        return pd.concat((pd.read_csv(f, **kwargs) for f in tqdm_wrap(filenames)), ignore_index=True)

def list_files_in_range(locations_or_visits,
                        min_id_prefix='00',
                        max_id_prefix='ff',
                        min_utc_hour='00', 
                        max_utc_hour='23',
                        utc_days=UTC_DAYS_TO_USE_IN_ANALYSIS):
    """
    Returns file paths within a range -- use for easy data loading. 
    Ranges are inclusive. (ie, include both min_id_prefix and max_id_prefix). 
    """
    assert min_id_prefix <= max_id_prefix
    assert min_utc_hour <= max_utc_hour
    assert locations_or_visits in ['locations', 'visits']
    assert min_id_prefix in VALID_ID_PREFIXES
    assert max_id_prefix in VALID_ID_PREFIXES
    for utc_day in utc_days:
        assert utc_day in VALID_UTC_DAYS
    assert min_utc_hour in VALID_UTC_HOURS
    assert max_utc_hour in VALID_UTC_HOURS

    all_files_in_range = []

    for id_prefix in VALID_ID_PREFIXES:
        if id_prefix < min_id_prefix or id_prefix > max_id_prefix:
            continue
        for utc_day in utc_days:
            for utc_hour in VALID_UTC_HOURS:
                if utc_hour < min_utc_hour or utc_hour > max_utc_hour:
                    continue
                file_directory, file_name = generate_filepath(locations_or_visits, id_prefix, utc_day, utc_hour)
                file_path = os.path.join(file_directory, file_name)
                all_files_in_range.append(file_path)

    print("Total files in range id_prefix %s-%s; utc_day %s-%s; utc_hour %s-%s: %i" % (
        min_id_prefix, 
        max_id_prefix,
        min(utc_days), 
        max(utc_days), 
        min_utc_hour, 
        max_utc_hour,
        len(all_files_in_range)))

    return all_files_in_range

def interpolate_locations_at_timestamps(datetime_strings, latitudes, longitudes):
    """
    Interpolate latitudes and longitudes each hour using linear interpolation. 
    
    Datetime strings should be UTC to avoid problems with daylight savings time. 
    """
    assert not dtype_pandas_series(datetime_strings)
    assert not dtype_pandas_series(latitudes)
    assert not dtype_pandas_series(longitudes)
    original_min_datetime = datetime.datetime.strptime(min(datetime_strings), DATETIME_FORMATTER_STRING)
    original_max_datetime = datetime.datetime.strptime(max(datetime_strings), DATETIME_FORMATTER_STRING)
    if (original_max_datetime - original_min_datetime).total_seconds() <= 3600:
        return [], [], []
    
    min_datetime = datetime.datetime(original_min_datetime.year, 
                                     original_min_datetime.month, 
                                     original_min_datetime.day,
                                     original_min_datetime.hour, 
                                     tzinfo=timezone.utc) + datetime.timedelta(hours=1)
    
    max_datetime = datetime.datetime(original_max_datetime.year, 
                                     original_max_datetime.month, 
                                     original_max_datetime.day,
                                     original_max_datetime.hour, 
                                     tzinfo=timezone.utc)
    timestamps_to_interpolate_at = np.arange(min_datetime.timestamp(), 
                                             max_datetime.timestamp() + 1, 
                                             3600)
    original_timestamps = np.array([convert_utc_datetime_string_to_timestamp(a) for a in datetime_strings])

    if not (
        (max(original_timestamps) >= max(timestamps_to_interpolate_at)) and
        (max(original_timestamps) <= (max(timestamps_to_interpolate_at) + 3600)) and
        (min(original_timestamps) <= min(timestamps_to_interpolate_at)) and
        (min(original_timestamps) >= (min(timestamps_to_interpolate_at) - 3600))):
        print("Warning! Timestamps are out of range in some way. Original timestamps were")
        print(original_min_datetime, original_max_datetime)
        assert False
    
    
    interpolation_latitude_f = interp1d(x=original_timestamps, y=latitudes)
    interpolation_longitude_f = interp1d(x=original_timestamps, y=longitudes)
    interpolated_latitudes = interpolation_latitude_f(timestamps_to_interpolate_at) 
    interpolated_longitudes = interpolation_longitude_f(timestamps_to_interpolate_at)
    
    return timestamps_to_interpolate_at, interpolated_latitudes, interpolated_longitudes

def convert_utc_datetime_string_to_timestamp(x):
    """
    This assumes x is a UTC string. 
    """
    as_utc_datetime = pytz.utc.localize(datetime.datetime.strptime(x, DATETIME_FORMATTER_STRING))
    return as_utc_datetime.timestamp()

def get_process_peak_memory_usage_kb():
    """Returns max memory ever used by the process, in KB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def is_notebook():
    """Returns whether python is running in a jupyter notebook."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def tqdm_wrap(*args, **kwargs):
    """Runs either tqdm() or tqdm_notebook() depending if we are in a notebook."""
    fn = tqdm.tqdm_notebook if is_notebook() else tqdm.tqdm
    return fn(*args, **kwargs)

def open_file_for_exclusive_writing(filename):
    """Opens the given file for exclusive writing.

    Returns None if the given file already exists and is nonempty,
    or if another process has opened the file for exclusive writing.

    This function can be used to distribute jobs among various processes.
    When a process wants to start working on a job, it should first open the
    output file for exclusive writing. This will only succeed if the file is
    empty and not opened by another process for exclusive writing. Jobs will
    thus work on disjoint sets of tasks.

    Note that flock() works over NFS, so this works even if the jobs are run
    on different machines. See https://manpages.debian.org/flock(2)
    """
    try:
        if os.stat(filename).st_size > 0:
            print('File is nonempty (early):', filename)
            return None
    except FileNotFoundError:
        pass
    f = open(filename, 'ab')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Another process has locked file:', filename)
        f.close()
        return None
    if f.tell() != 0:
        print('File is nonempty:', filename)
        f.close()
        return None
    return f

