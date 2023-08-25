import json
from constants_and_util import *
import math
from dask import dataframe as dd 
from dask.diagnostics import ProgressBar
import dask
import fiona
from shapely.geometry import shape, Point, LineString, Polygon
from shapely import wkt
import pandas as pd
from scipy.stats import scoreatpercentile
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import numpy as np
import sys
import pickle
from IPython import embed
import itertools
import time
import copy
import random
import pytz
from timezonefinder import TimezoneFinder
from scipy.interpolate import interp1d
import zipfile
import io
from scipy.spatial import cKDTree
import geohash as geohash_lib
from collections import Counter, defaultdict
import xmltodict
import reverse_geocoder
import requests
import json
from geopy.geocoders import Nominatim, GoogleV3, Photon, AzureMaps, GeoNames, What3Words, MapBox
from traceback import print_exc
import matplotlib.pyplot as plt
import warnings
import hashlib
import csv
import glob
from geopandas.tools import sjoin
import os
import shutil
import argparse
import geopandas

def run_all_jobs_for_extract_middle_of_night_locations():
    for id_prefix in VALID_ID_PREFIXES:
        outfile_name = get_middle_of_night_locations_outfile(id_prefix)
        os.makedirs(os.path.dirname(outfile_name), exist_ok=True)
        outfile = open_file_for_exclusive_writing(outfile_name)
        if outfile is None: continue # No need to run job.
        with outfile: # The 'with' will close the file when done.
            extract_middle_of_night_locations(id_prefix=id_prefix, outfile=outfile,
                                            **MIDDLE_OF_NIGHT_HYPERPARAMS)

def get_middle_of_night_locations_outfile(id_prefix):
    """
    Gives the path for the middle of night locations for a given id prefix. 
    This is grouped by id_prefix. 
    We save inferred home locations for each user (as a pickle).

    annotations_string is a string which we concatenate onto the filename so we don't overwrite previous files as we do data processing. 
    """
    return os.path.join(MIDDLE_OF_NIGHT_LOCATIONS_DIR, 'inferred_home_locations_id_prefix_%s.pkl' % id_prefix)

def extract_middle_of_night_locations(id_prefix, 
                                      outfile,
                                      local_start_hour, 
                                      local_end_hour, 
                                      weekdays_only, 
                                      min_nights_required,
                                      max_distance, 
                                      min_frac_near_median,
                                      min_total_pings_a_user_must_have):

    """
    Method for inferring home location of each user based on where they are at night. 

    Parameters for filtering users with reliable locations (We will likely have to change these.)
    local_start_hour,local_end_hour range is inclusive, specifies what times we look at as "middle of night locations".  
    weekdays_only: only look at Mon-Fri for middle of night locations. 
    min_nights_required, max_distance, min_frac_near_median: hyperparameters for new_infer_home_or_work_location_for_single_user. 
    min_total_pings_a_user_must_have: only look at users with this number of pings. 
    write_out_middle_of_night_locations: if false, do not write out the inferred locations (if we're just testing different hyperparameter settings, for example) 
    """
    assert type(local_start_hour) is int
    assert type(local_end_hour) is int

    local_start_before_noon = local_start_hour < 12
    local_end_before_noon = local_end_hour < 12
    valid_days_and_hours = set() # Filter for hours/days in the range we are looking for. This involves some annoying case-by-case code. 
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    all_days = weekdays + ['Saturday' ,'Sunday']

    if local_start_before_noon and local_end_before_noon:
        assert local_start_hour < local_end_hour
        if weekdays_only:
            days_to_use = weekdays
        else:
            days_to_use = all_days

        for hour in range(local_start_hour, local_end_hour + 1):
            for day in days_to_use:
                valid_days_and_hours.add('%s %02d' % (day, hour))
    elif (not local_start_before_noon) and local_end_before_noon:
        if weekdays_only:
            for hour in range(local_start_hour, 24):
                for day in ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday']:
                    valid_days_and_hours.add('%s %02d' % (day, hour))
            for hour in range(0, local_end_hour + 1):
                for day in weekdays:
                    valid_days_and_hours.add('%s %02d' % (day, hour))
        else:
            for hour in range(local_start_hour, 24):
                for day in all_days:
                    valid_days_and_hours.add('%s %02d' % (day, hour))
            for hour in range(0, local_end_hour + 1):
                for day in all_days:
                    valid_days_and_hours.add('%s %02d' % (day, hour))
    else:
        raise Exception("Something is weird here.")
    print("Valid weekdays and hours:\n%s" % '\n'.join(sorted(list(valid_days_and_hours))))  
    
    all_outfiles = list_files_in_range('locations', 
                                       min_id_prefix=id_prefix, 
                                       max_id_prefix=id_prefix,
                                       utc_days=UTC_DAYS_TO_USE_IN_ANALYSIS)

    all_dataframes = []
    total_pings_by_user = {} # we are only interested in people who have a minimum number of pings. 
    for filename in all_outfiles:
        d = load_ping_file(file_path=filename, filter_horizontal_acc=False)
        assert d['safegraph_id'].map(lambda x:x[:2] == id_prefix).all()

        pings_by_user = Counter(d['safegraph_id'])
        for user in pings_by_user:
            if user not in total_pings_by_user:
                total_pings_by_user[user] = pings_by_user[user]
            else:
                total_pings_by_user[user] += pings_by_user[user]  

        d['local_weekday_and_hour'] = d['local_datetime'].map(
            lambda x:datetime.datetime.strptime(x, DATETIME_FORMATTER_STRING).strftime('%A %H'))

        old_len = len(d)

        d = d.loc[d['local_weekday_and_hour'].map(lambda x:x in valid_days_and_hours)] # filter for middle-of-night pings. 


        def get_day_to_group_by(x):
            as_datetime = datetime.datetime.strptime(x, DATETIME_FORMATTER_STRING)
            if as_datetime.hour < 12:
                datetime_to_group_by = as_datetime
            else:
                datetime_to_group_by = datetime.datetime(as_datetime.year, as_datetime.month, as_datetime.day) + datetime.timedelta(days=1)
            return datetime_to_group_by.strftime('%Y-%m-%d')

        print("original length of %s is %i; after filtering for middle-of-night locs, %i; users in this dataframe, %i, total users in all dataframes seen thus far, %i" % (
            filename, old_len, len(d), len(pings_by_user), len(total_pings_by_user)))
        d['day_to_group_by'] = d['local_datetime'].map(get_day_to_group_by)
        if len(d) > 0:
            all_dataframes.append(d[['safegraph_id', 
                'latitude', 
                'longitude', 
                'day_to_group_by', 
                'horizontal_accuracy', 
                'utc_datetime', 
                'local_datetime']])
    
    n_total_users = len(total_pings_by_user)
    users_with_enough_pings_to_try_to_infer_homes = set([a for a in total_pings_by_user if total_pings_by_user[a] >= min_total_pings_a_user_must_have])
    n_users_with_enough_pings_to_try_to_infer_homes = len(users_with_enough_pings_to_try_to_infer_homes)
    print("Number of users with the required number of pings: %i/%i" % 
        (n_users_with_enough_pings_to_try_to_infer_homes, n_total_users))

    combined_df = pd.concat(all_dataframes)
    combined_df = combined_df.loc[combined_df['safegraph_id'].map(lambda x:x in users_with_enough_pings_to_try_to_infer_homes)]
    combined_df.index = range(len(combined_df))

    grouped_d = combined_df.groupby('safegraph_id')
    users_with_home_locations = {}
    filter_out_reasons = {}
    for safegraph_id, user_d in grouped_d:
        filter_out_reason, data_to_save = new_infer_home_or_work_location_for_single_user(
            user_d, 
            min_nights_required=min_nights_required,
            max_distance=max_distance,
            min_frac_near_median=min_frac_near_median)
        if filter_out_reason is None:
            users_with_home_locations[safegraph_id] = data_to_save
        else:
            if filter_out_reason not in filter_out_reasons:
                filter_out_reasons[filter_out_reason] = 0
            filter_out_reasons[filter_out_reason] += 1

    print("Total number of user-night observations: %i" % len(combined_df))
    print("Was able to infer locations for %i users. Reasons for removing users are:" % (len(users_with_home_locations)),
        filter_out_reasons)
    pickle.dump(users_with_home_locations, outfile)

def new_infer_home_or_work_location_for_single_user(d, 
                                            min_nights_required, 
                                            max_distance, 
                                            min_frac_near_median, 
                                            min_nights_near_median=0,
                                            clustering_algorithm_for_centroid=None, 
                                            clustering_kwargs=None):
    """
    Given a dataframe of pings for a single user, infer home or work location. 
    This is the inner method in home/work location inference methods: it doesn't do things like filter for pings during workplace hours. 
    Rather, it looks for clusters of pings for a single user. 

    Return two arguments: an error message if home/work cannot be inferred (None if it can)
    and a dictionary of information about the home if it can be inferred (None otherwise)

    basic idea: interpolate user's location at each hour, filter for hours where they don't move, take median of those hours. 

    Parameters: 

    min_nights_required: we need data from at least this many nights. Note that we only count a night if the user also has an interpolated hour which is stationary. 
    max_distance: we use this to determine what constitutes too much movement between hours, and also when locations are too scattered around the median. 
    min_frac_near_median: at least this proportion of hours must be near the median. 

    Because this was originally designed to infer home locations, some of the naming is a little confusing -- it uses "nights" when actually it means to refer to "days or nights". In general, it refers to distinct days -- eg, 9 AM-5 PM periods for workplace inference, and 1-5 AM periods for home inference. 
    """
    clustering_kwargs = copy.deepcopy(clustering_kwargs)
    if len(set(d['day_to_group_by'])) < min_nights_required:
        return 'too_few_nights', None
    
    grouped_by_night = d.groupby('day_to_group_by')
    
    all_good_lats = []
    all_good_lons = []
    all_groups = [] # what day does each interpolated ping occur on. 
    n_nights = 0
    
    for night, night_d in grouped_by_night:
        interpolated_timestamps, interpolated_latitudes, interpolated_longitudes = interpolate_locations_at_timestamps(
            datetime_strings=night_d['utc_datetime'].values, 
            latitudes=night_d['latitude'].values, 
            longitudes=night_d['longitude'].values)
        if len(interpolated_timestamps) > 1:
            dists = compute_distance_between_two_lat_lons(lat1=interpolated_latitudes[1:], 
                                                     lon1=interpolated_longitudes[1:], 
                                                     lat2=interpolated_latitudes[:-1], 
                                                     lon2=interpolated_longitudes[:-1]) 
            dists = np.array(list(dists) + [np.inf]) # automatically drop last interpolated timestamp. 
            good_lats = interpolated_latitudes[dists < max_distance]
            good_lons = interpolated_longitudes[dists < max_distance]
            if len(good_lons) > 0:
                n_nights += 1
                all_good_lats += list(good_lats)
                all_good_lons += list(good_lons)
                all_groups += [night for i in range(len(good_lons))]
    if n_nights < min_nights_required:
        return 'too_few_nights', None
    all_good_lats = np.array(all_good_lats)
    all_good_lons = np.array(all_good_lons)
    all_groups = np.array(all_groups)

    if clustering_algorithm_for_centroid is not None:
        x, y, z = lonlat_to_xyz(lon=all_good_lons, lat=all_good_lats)
        n_unique_locs = len(pd.DataFrame({'a':all_good_lons, 'b':all_good_lats}).drop_duplicates())
        spatial_locs = np.array([x, y, z]).transpose()
        if clustering_algorithm_for_centroid == 'kmeans':
            clustering_kwargs['n_clusters'] = min(clustering_kwargs['n_clusters'], n_unique_locs)
            clustering_model = KMeans(random_state=0, **clustering_kwargs)
        elif clustering_algorithm_for_centroid  == 'DBSCAN':
            clustering_model = DBSCAN(min_samples=3, **clustering_kwargs)
        elif clustering_algorithm_for_centroid == 'hierarchical_clustering':
            clustering_kwargs['n_clusters'] = min(clustering_kwargs['n_clusters'], n_unique_locs)
            clustering_model = AgglomerativeClustering(linkage='single', **clustering_kwargs)
        else:
            raise Exception("Invalid clustering algorithm")
        clustering_model.fit(spatial_locs)
        clustering_labels = clustering_model.labels_
        cluster_counts = Counter(clustering_labels)
        largest_cluster_idx = cluster_counts.most_common()[0][0]
        largest_cluster_members =  clustering_labels == largest_cluster_idx
        median_lat = np.median(all_good_lats[largest_cluster_members]) # Convert back to lat,lon space by taking the median. 
        median_lon = np.median(all_good_lons[largest_cluster_members])
    else:
        median_lat = np.median(all_good_lats)
        median_lon = np.median(all_good_lons)
    
    distance_from_median = compute_distance_between_two_lat_lons(lat1=median_lat, 
                                                            lon1=median_lon, 
                                                            lat2=all_good_lats, 
                                                            lon2=all_good_lons)

    not_outliers = (distance_from_median < max_distance)
    n_nights_near_median = len(set(all_groups[not_outliers])) # in case we want to filter for people who return to the place on several unique days. 
 
    frac_near_median = not_outliers.mean()
    
    if frac_near_median < min_frac_near_median:
        return 'too_scattered', None

    if n_nights_near_median < min_nights_near_median:
        return 'too_few_nights_near_median', None

    all_good_lats = all_good_lats[not_outliers]
    all_good_lons = all_good_lons[not_outliers]
    median_lat = np.median(all_good_lats)
    median_lon = np.median(all_good_lons)

    lat_std_err = np.std(all_good_lats, ddof=1) / np.sqrt(len(all_good_lats)) # this is very heuristic. 
    lon_std_err = np.std(all_good_lons, ddof=1) / np.sqrt(len(all_good_lons))

    std_err_in_meters = compute_distance_between_two_lat_lons(lat1=median_lat,
        lat2=median_lat + lat_std_err, 
        lon1=median_lon,
        lon2=median_lon + lon_std_err)
    
    results_to_return = {'inferred_home_location_lat':median_lat, 
    'inferred_home_location_lon':median_lon, 
    'n_nights_used_to_infer_including_outliers':n_nights, 
    'n_hourly_pings_used_to_infer_including_outliers':len(not_outliers), 
    'n_hourly_pings_used_to_infer_not_including_outliers':not_outliers.sum(),
    'frac_near_median':frac_near_median,
    'std_err_in_meters':std_err_in_meters, 
    'n_nights_went_near_median':n_nights_near_median}

    return None, results_to_return

class CensusBlockGroups:
    """
    A class for loading geographic and demographic data from the ACS. 

    A census block group is a relatively small area (I think it's a couple hundred households). 
    Less good than houses but still pretty granular. https://en.wikipedia.org/wiki/Census_block_group

    Data was downloaded from https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-data.html
    We use the most recent ACS 5-year estimates: 2013-2017, eg: 
    wget https://www2.census.gov/geo/tiger/TIGER_DP/2017ACS/ACS_2017_5YR_BG.gdb.zip
    These files are convenient because they combine both geographic boundaries + demographic data, leading to a cleaner join. 

    Documentation of column names is at /projects/p30458/non_safegraph_datasets/census_block_group_data/ACS_5_year_2013_to_2017_joined_to_blockgroup_shapefiles/documentation_ACS_2017_joined_to_blockgroup_shapefile.txt

    The main method for data access is get_demographic_stats_of_point. Sample usage: 
    x = CensusBlockGroups(gdb_files=['ACS_2017_5YR_BG_51_VIRGINIA.gdb'])
    x.get_demographic_stats_of_points(latitudes=[38.8816], longitudes=[-77.0910], desired_cols=['p_black', 'p_white', 'mean_household_income'])
    """
    def __init__(self, base_directory=BASE_DIRECTORY_FOR_EVERYTHING, 
        gdb_files=None, 
        county_to_msa_mapping_file='august_2017_county_to_metropolitan_mapping.csv'):
        self.base_directory = base_directory
        if gdb_files is None:
            self.gdb_files = ['ACS_2017_5YR_BG.gdb']
        else:
            self.gdb_files = gdb_files
        self.crs_to_use = WGS_84_CRS # https://epsg.io/4326, WGS84 - World Geodetic System 1984, used in GPS. 
        self.county_to_msa_mapping_file = county_to_msa_mapping_file
        self.load_raw_dataframes() # Load in raw geometry and demographic dataframes. 
        
        self.annotate_with_race()
        self.annotate_with_education()
        self.annotate_with_income()
        self.annotate_with_rent_as_percentage_of_block_group_income()
        self.annotate_with_counties_to_msa_mapping()

    def annotate_with_race(self):
        """
        Note that the Experienced Segregation paper only considers black/white.
        B03002e1  HISPANIC OR LATINO ORIGIN BY RACE: Total: Total population -- (Estimate)
        B03002e3  HISPANIC OR LATINO ORIGIN BY RACE: Not Hispanic or Latino: White alone: Total population -- (Estimate)
        B03002e4  HISPANIC OR LATINO ORIGIN BY RACE: Not Hispanic or Latino: Black or African American alone: Total population -- (Estimate)
        B03002e6  HISPANIC OR LATINO ORIGIN BY RACE: Not Hispanic or Latino: Asian alone: Total population -- (Estimate)
        B03002e12 HISPANIC OR LATINO ORIGIN BY RACE: Hispanic or Latino: Total population -- (Estimate)
        """
        print("annotating with race")
        self.block_group_d['p_black'] = self.block_group_d['B03002e4'] / self.block_group_d['B03002e1']
        self.block_group_d['p_white'] = self.block_group_d['B03002e3'] / self.block_group_d['B03002e1']
        self.block_group_d['p_asian'] = self.block_group_d['B03002e6'] / self.block_group_d['B03002e1']
        self.block_group_d['p_hispanic'] = self.block_group_d['B03002e12'] / self.block_group_d['B03002e1']
        print(self.block_group_d[['p_black', 'p_white', 'p_asian', 'p_hispanic']].describe())

    def annotate_with_education(self):
        keys = [f'B15003e{i}' for i in range(2, 19)]
        high_school_or_lower = self.block_group_d[keys].sum(axis=1)
        total = self.block_group_d['B15003e1']
        self.block_group_d['p_high_school_or_lower'] = high_school_or_lower / total
        print(self.block_group_d[['p_high_school_or_lower']].describe())

    def load_raw_dataframes(self):
        """
        Read in the original demographic + geographic data. 
        """
        self.block_group_d = None
        self.geometry_d = None
        demographic_layer_names = ['X25_HOUSING_CHARACTERISTICS', 'X01_AGE_AND_SEX', 'X03_HISPANIC_OR_LATINO_ORIGIN', 'X19_INCOME', 'X15_EDUCATIONAL_ATTAINMENT']
        for file in self.gdb_files: 
            full_path = os.path.join(self.base_directory, file)
            layer_list = fiona.listlayers(full_path)
            print(file)
            print(layer_list)
            geographic_layer_name = [a for a in layer_list if a[:15] == 'ACS_2017_5YR_BG']
            assert len(geographic_layer_name) == 1
            geographic_layer_name = geographic_layer_name[0]

            geographic_data = geopandas.read_file(full_path, layer=geographic_layer_name).to_crs(self.crs_to_use)
            print(geographic_data.columns)
            geographic_data = geographic_data.sort_values(by='GEOID_Data')[['GEOID_Data', 'geometry', 'STATEFP', 'COUNTYFP', 'TRACTCE']]
            for demographic_idx, demographic_layer_name in enumerate(demographic_layer_names):
                assert demographic_layer_name in layer_list
                if demographic_idx == 0:
                    demographic_data = geopandas.read_file(full_path, layer=demographic_layer_name)
                else:
                    old_len = len(demographic_data)
                    new_df = geopandas.read_file(full_path, layer=demographic_layer_name)
                    assert sorted(new_df['GEOID']) == sorted(demographic_data['GEOID'])
                    demographic_data = demographic_data.merge(new_df, on='GEOID', how='inner')
                    assert old_len == len(demographic_data)
            demographic_data = demographic_data.sort_values(by='GEOID')

            shared_geoids = set(demographic_data['GEOID'].values).intersection(set(geographic_data['GEOID_Data'].values))
            print("Length of demographic data: %i; geographic data %i; %i GEOIDs in both" % (len(demographic_data), len(geographic_data), len(shared_geoids)))
            
            demographic_data = demographic_data.loc[demographic_data['GEOID'].map(lambda x:x in shared_geoids)]
            geographic_data = geographic_data.loc[geographic_data['GEOID_Data'].map(lambda x:x in shared_geoids)]
            
            demographic_data.index = range(len(demographic_data))
            geographic_data.index = range(len(geographic_data))

            assert (geographic_data['GEOID_Data'] == demographic_data['GEOID']).all()
            assert len(geographic_data) == len(set(geographic_data['GEOID_Data']))


            if self.block_group_d is None:
                self.block_group_d = demographic_data
            else:
                self.block_group_d = pd.concat([self.block_group_d, demographic_data])

            if self.geometry_d is None:
                self.geometry_d = geographic_data
            else:
                self.geometry_d = pd.concat([self.geometry_d, geographic_data])

        assert pd.isnull(self.geometry_d['STATEFP']).sum() == 0
        good_idxs = self.geometry_d['STATEFP'].map(lambda x:x in FIPS_CODES_FOR_50_STATES_PLUS_DC).values
        print("Warning: the following State FIPS codes are being filtered out")
        print(self.geometry_d.loc[~good_idxs, 'STATEFP'].value_counts())
        print("%i/%i Census Block Groups in total removed" % ((~good_idxs).sum(), len(good_idxs)))
        self.geometry_d = self.geometry_d.loc[good_idxs]
        self.block_group_d = self.block_group_d.loc[good_idxs]
        self.geometry_d.index = self.geometry_d['GEOID_Data'].values
        self.block_group_d.index = self.block_group_d['GEOID'].values
 
    def annotate_with_income(self):
        """
        We want a single income number for each block group. This method computes that. 
        """ 
        print("Computing household income")
        codebook_string = """
        B19001e2    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): Less than $10,000: Households -- (Estimate)
        B19001e3    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $10,000 to $14,999: Households -- (Estimate)
        B19001e4    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $15,000 to $19,999: Households -- (Estimate)
        B19001e5    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $20,000 to $24,999: Households -- (Estimate)
        B19001e6    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $25,000 to $29,999: Households -- (Estimate)
        B19001e7    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $30,000 to $34,999: Households -- (Estimate)
        B19001e8    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $35,000 to $39,999: Households -- (Estimate)
        B19001e9    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $40,000 to $44,999: Households -- (Estimate)
        B19001e10   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $45,000 to $49,999: Households -- (Estimate)
        B19001e11   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $50,000 to $59,999: Households -- (Estimate)
        B19001e12   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $60,000 to $74,999: Households -- (Estimate)
        B19001e13   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $75,000 to $99,999: Households -- (Estimate)
        B19001e14   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $100,000 to $124,999: Households -- (Estimate)
        B19001e15   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $125,000 to $149,999: Households -- (Estimate)
        B19001e16   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $150,000 to $199,999: Households -- (Estimate)
        B19001e17   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $200,000 or more: Households -- (Estimate)
        """
        self.income_bin_edges = [0] + list(range(10000, 50000, 5000)) + [50000, 60000, 75000, 100000, 125000, 150000, 200000]

        income_column_names_to_vals = {}
        column_codes = codebook_string.split('\n')
        for f in column_codes:
            if len(f.strip()) == 0:
                continue
            col_name = f.split('HOUSEHOLD INCOME')[0].strip()
            if col_name == 'B19001e2':
                val = 10000
            elif col_name == 'B19001e17':
                val = 200000
            else:
                lower_bound = float(f.split('$')[1].split()[0].replace(',', ''))
                upper_bound = float(f.split('$')[2].split(':')[0].replace(',', ''))
                val = (lower_bound + upper_bound) / 2
            income_column_names_to_vals[col_name] = val
            print("The value for column %s is %2.1f" % (col_name, val))

        self.block_group_d['total_household_income'] = 0.
        self.block_group_d['total_households'] = 0.
        for col in income_column_names_to_vals:
            self.block_group_d['total_household_income'] += self.block_group_d[col] * income_column_names_to_vals[col]
            self.block_group_d['total_households'] += self.block_group_d[col]
        self.block_group_d['mean_household_income'] = 1.*self.block_group_d['total_household_income'] / self.block_group_d['total_households']   
        self.block_group_d['median_household_income'] = self.block_group_d['B19013e1'] # MEDIAN HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): Median household income in the past 12 months (in 2017 inflation-adjusted dollars): Households -- (Estimate)
        assert (self.block_group_d['total_households'] == self.block_group_d['B19001e1']).all() # sanity check: our count should agree with theirs.
        assert (pd.isnull(self.block_group_d['mean_household_income']) == (self.block_group_d['B19001e1'] == 0)).all()
        print("Warning: missing income data for %2.1f%% of census blocks with 0 households" % (pd.isnull(self.block_group_d['mean_household_income']).mean() * 100))
        self.income_column_names_to_vals = income_column_names_to_vals
        assert len(self.income_bin_edges) == len(self.income_column_names_to_vals)
        print(self.block_group_d[['mean_household_income', 'total_households']].describe())

    def annotate_with_counties_to_msa_mapping(self):
        """
        Annotate with metropolitan area info for consistency with Experienced Segregation paper. 
        """
        print("Loading county to MSA mapping")
        self.counties_to_msa_df = pd.read_csv(os.path.join(self.base_directory, self.county_to_msa_mapping_file), skiprows=2, dtype={'FIPS State Code':str, 'FIPS County Code':str})
        print("%i rows read" % len(self.counties_to_msa_df))
        self.counties_to_msa_df = self.counties_to_msa_df[['CBSA Title', 
                                                           'Metropolitan/Micropolitan Statistical Area', 
                                                           'State Name',
                                                           'FIPS State Code', 
                                                           'FIPS County Code']]                                             

        self.counties_to_msa_df.columns = ['CBSA Title', 
                                           'Metropolitan/Micropolitan Statistical Area', 
                                           'State Name',
                                           'STATEFP', 
                                           'COUNTYFP']

        self.counties_to_msa_df = self.counties_to_msa_df.dropna(how='all') # remove a couple blank rows. 
        assert self.counties_to_msa_df['Metropolitan/Micropolitan Statistical Area'].map(lambda x:x in ['Metropolitan Statistical Area', 'Micropolitan Statistical Area']).all()
        print("Number of unique Metropolitan statistical areas: %i" % 
            len(set(self.counties_to_msa_df.loc[self.counties_to_msa_df['Metropolitan/Micropolitan Statistical Area'] == 'Metropolitan Statistical Area', 'CBSA Title'])))
        print("Number of unique Micropolitan statistical areas: %i" % 
            len(set(self.counties_to_msa_df.loc[self.counties_to_msa_df['Metropolitan/Micropolitan Statistical Area'] == 'Micropolitan Statistical Area', 'CBSA Title'])))
        old_len = len(self.geometry_d)
        assert len(self.counties_to_msa_df.drop_duplicates(['STATEFP', 'COUNTYFP'])) == len(self.counties_to_msa_df)

        
        self.geometry_d = self.geometry_d.merge(self.counties_to_msa_df, 
                                                on=['STATEFP', 'COUNTYFP'], 
                                                how='left')
        self.geometry_d.index = self.geometry_d['GEOID_Data'].values

        assert len(self.geometry_d) == old_len
        assert (self.geometry_d.index == self.block_group_d.index).all()
        
    def annotate_with_rent_as_percentage_of_block_group_income(self):
        print("Computing monthly rent -> annual income multiplier")
        """
        B25070e1    GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME IN THE PAST 12 MONTHS: Total: Renter-occupied housing units -- (Estimate)
        B25070e2    GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME IN THE PAST 12 MONTHS: Less than 10.0 percent: Renter-occupied housing units -- (Estimate)
        B25070e3    GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME IN THE PAST 12 MONTHS: 10.0 to 14.9 percent: Renter-occupied housing units -- (Estimate)
        B25070e4    GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME IN THE PAST 12 MONTHS: 15.0 to 19.9 percent: Renter-occupied housing units -- (Estimate)
        B25070e5    GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME IN THE PAST 12 MONTHS: 20.0 to 24.9 percent: Renter-occupied housing units -- (Estimate)
        B25070e6    GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME IN THE PAST 12 MONTHS: 25.0 to 29.9 percent: Renter-occupied housing units -- (Estimate)
        B25070e7    GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME IN THE PAST 12 MONTHS: 30.0 to 34.9 percent: Renter-occupied housing units -- (Estimate)
        B25070e8    GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME IN THE PAST 12 MONTHS: 35.0 to 39.9 percent: Renter-occupied housing units -- (Estimate)
        B25070e9    GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME IN THE PAST 12 MONTHS: 40.0 to 49.9 percent: Renter-occupied housing units -- (Estimate)
        B25070e10   GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME IN THE PAST 12 MONTHS: 50.0 percent or more: Renter-occupied housing units -- (Estimate)
        B25070e11   GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME IN THE PAST 12 MONTHS: Not computed: Renter-occupied housing units -- (Estimate)
        """
        cols_to_vals = {'B25070e2':0.1, 
                        'B25070e3':0.125, 
                        'B25070e4':0.175, 
                        'B25070e5':0.225, 
                        'B25070e6':0.275, 
                        'B25070e7':0.325, 
                        'B25070e8':0.375, 
                        'B25070e9':0.45, 
                        'B25070e10':0.50}
        not_computed = self.block_group_d['B25070e11'] / self.block_group_d['B25070e1']
        print('Median fraction of people lacking data: %2.3f' % np.median(not_computed.values[self.block_group_d['B25070e1'] > 0]))
        
        total_count = self.block_group_d['B25070e1'].values - self.block_group_d['B25070e11'].values
        multiplier = 0
        fracs = 0
        for k in cols_to_vals:
            multiplier = (multiplier + 
                          12 * # number of months
                          (1 / cols_to_vals[k]) * # how much monthly income monthly rent implies
                          (self.block_group_d[k].values/total_count) # fraction of households
                         )
            fracs = fracs + self.block_group_d[k].values/total_count
        assert np.isclose(fracs[total_count > 0], 1).all()
            
        self.block_group_d['monthly_rent_to_annual_income_multiplier'] = multiplier
        assert ((total_count == 0) == np.isnan(multiplier)).all()                                                                
        print("Median n: %2.3f; mean n %2.3f" % (
            np.median(total_count), 
            np.mean(total_count)))

        self.block_group_d['median_monthly_rent_to_annual_income_multiplier'] = 12. / (self.block_group_d['B25071e1'].values / 100.) # This returns the median. 
        print(self.block_group_d[['monthly_rent_to_annual_income_multiplier', 'median_monthly_rent_to_annual_income_multiplier']].describe())

    def get_demographic_stats_of_points(self, latitudes, longitudes, desired_cols):
        """
        Given a list or array of latitudes and longitudes, matches to Census Block Group. 
        Returns a dictionary which includes the state and county FIPS code, along with any columns in desired_cols. 

        This method assumes the latitudes and longitudes are in https://epsg.io/4326, which is what I think is used for Android/iOS -> SafeGraph coordinates. 
        """
        assert not dtype_pandas_series(latitudes)
        assert not  dtype_pandas_series(longitudes)
        assert len(latitudes) == len(longitudes)

        t0 = time.time()

        start_idx = 0
        end_idx = start_idx + int(1e6)
        merged = []
        while start_idx < len(longitudes):
            print("Doing spatial join on points with indices from %i-%i" % (start_idx, min(end_idx, len(longitudes))))

            points = geopandas.GeoDataFrame(pd.DataFrame({'placeholder':np.array(range(start_idx, min(end_idx, len(longitudes))))}), # this column doesn't matter. We just have to create a geo data frame. 
                geometry=geopandas.points_from_xy(longitudes[start_idx:end_idx], latitudes[start_idx:end_idx]),
                crs=self.crs_to_use) 
            merged.append(sjoin(points, self.geometry_d[['geometry']], how='left', op='within'))
            assert len(merged[-1]) == len(points)
            start_idx += int(1e6)
            end_idx += int(1e6)
        merged = pd.concat(merged)
        merged.index = range(len(merged))
        assert list(merged.index) == list(merged['placeholder'])

        could_not_match = pd.isnull(merged['index_right']).values
        print("Cannot match to a CBG for a fraction %2.3f of points" % could_not_match.mean())

        results = {}
        for k in desired_cols + ['state_fips_code', 'county_fips_code', 'Metropolitan/Micropolitan Statistical Area', 'CBSA Title', 'GEOID_Data', 'TRACTCE']:
            results[k] = [None] * len(latitudes)
        results = pd.DataFrame(results)
        matched_geoids = merged['index_right'].values[~could_not_match]
        for c in desired_cols:
            results.loc[~could_not_match, c] = self.block_group_d.loc[matched_geoids, c].values
            if c in ['p_white', 'p_black', 'mean_household_income', 'median_household_income', 'new_census_monthly_rent_to_annual_income_multiplier', 'new_census_median_monthly_rent_to_annual_income_multiplier']:
                results[c] = results[c].astype('float')

        results.loc[~could_not_match, 'state_fips_code'] = self.geometry_d.loc[matched_geoids, 'STATEFP'].values
        results.loc[~could_not_match, 'county_fips_code'] = self.geometry_d.loc[matched_geoids, 'COUNTYFP'].values
        results.loc[~could_not_match, 'Metropolitan/Micropolitan Statistical Area'] = self.geometry_d.loc[matched_geoids,'Metropolitan/Micropolitan Statistical Area'].values
        results.loc[~could_not_match, 'CBSA Title'] = self.geometry_d.loc[matched_geoids, 'CBSA Title'].values
        results.loc[~could_not_match, 'GEOID_Data'] = self.geometry_d.loc[matched_geoids, 'GEOID_Data'].values
        results.loc[~could_not_match, 'TRACTCE'] = self.geometry_d.loc[matched_geoids, 'TRACTCE'].values

        print("Total query time is %2.3f" % (time.time() - t0))
        return results

def home_locations_for_id_prefix_path(id_prefix):
    """
    Get the path for home locations for a single ID prefix. 
    """
    assert id_prefix in VALID_ID_PREFIXES
    return os.path.join(HOME_LOCATIONS_DIR, 'home_locations_id_prefix_%s.csv.gz' % id_prefix)

def create_final_zestimate_files(new_cbg_data=None):
    d = []
    for id_prefix in VALID_ID_PREFIXES:
        pf = pickle.load(open(get_middle_of_night_locations_outfile(id_prefix), 'rb'))
        for safegraph_id, row in pf.items():
            d.append({'safegraph_id': safegraph_id, **row})
    d = pd.DataFrame(d)

    zillow_df = pd.read_csv(ZILLOW_QUERY_RESULTS_FILE)
    d = d.merge(zillow_df, how='inner', on=['inferred_home_location_lat', 'inferred_home_location_lon'])

    ses_col_to_use = 'rent_zestimate' # insist everyone has non null values for this. 
    if new_cbg_data is None:
        new_cbg_data = CensusBlockGroups()
    zestimates = load_filtered_zestimate_data(d=d,
                                            zestimate_truncation_threshold=1e7, 
                                            rent_zestimate_truncation_threshold=2e4,
                                            zillow_distance_threshold_in_meters=100, 
                                            corelogic_distance_threshold_in_meters=100, 
                                            max_count_by_location_for_single_family_residence=10, 
                                            ses_col_to_use=ses_col_to_use, 
                                            new_cbg_data=new_cbg_data)
    zestimates = zestimates.rename(columns={'inferred_home_location_lat':'inferred_home_location_lat_DO_NOT_USE', 
                                            'inferred_home_location_lon':'inferred_home_location_lon_DO_NOT_USE'})
    zestimates['id_prefix'] = zestimates['safegraph_id'].str.slice(stop=2)
    for id_prefix, df in zestimates.groupby('id_prefix'):
        outfile_name = home_locations_for_id_prefix_path(id_prefix)
        os.makedirs(os.path.dirname(outfile_name), exist_ok=True)
        df.drop(columns='id_prefix').to_csv(outfile_name, compression='gzip')

def load_filtered_zestimate_data(d, zestimate_truncation_threshold, rent_zestimate_truncation_threshold, zillow_distance_threshold_in_meters, corelogic_distance_threshold_in_meters, max_count_by_location_for_single_family_residence, ses_col_to_use, new_cbg_data):

    """
    Given paths zestimates_filepaths for the Zestimates data, load into a single dataframe after applying various filters and sanity checks. 
    Also annotate with Census data. 

    Arguments: 
    zestimate_truncation_threshold: If a Zestimate is greater than this, truncate to this number. 
    zillow_distance_threshold_in_meters: remove zestimates further than this many meters from the original SafeGraph lat,lon. 
    corelogic_distance_threshold_in_meters: remove Zestimates where the CoreLogic match is further than this from the original SafeGraph lat,lon. 
    max_count_by_location_for_single_family_residence: remove single family addresses with more than this many lat,lons matched to them. If None, don't filter out addresses. 
    ses_col_to_use: should be rent_zestimate or zestimate. 
    new_cbg_data: a CensusBlockGroup data structure, used for annotating with Census data. 

    These are not the only filters we apply; they are just the filters which require parameters. 
    """
    assert d['safegraph_id'].duplicated().sum() == 0
    print("Total number loaded: %i" % len(d))
    assert ses_col_to_use in ['rent_zestimate', 'zestimate']

    print("Linking to new Census data")
    link_to_new_census_data = new_cbg_data.get_demographic_stats_of_points(
        latitudes=d['zillow_lat'].values, 
        longitudes=d['zillow_lon'].values,
        desired_cols=['p_black', 'p_white', 'p_asian', 'p_hispanic', 'p_high_school_or_lower', 'mean_household_income', 'median_household_income', 'monthly_rent_to_annual_income_multiplier', 'median_monthly_rent_to_annual_income_multiplier'])
    for k in link_to_new_census_data:
        d['new_census_%s' % k] = link_to_new_census_data[k].values

    for c in d.columns:
        col_null = pd.isnull(d[c])
        print('%-60s is null %2.5f of the time' % (c, col_null.mean()))

    duplicate_lat_lons = d[['inferred_home_location_lat', 'inferred_home_location_lon']].duplicated(keep=False)
    print("fraction %2.5f of rows have duplicate lat,lons; removing" % duplicate_lat_lons.mean())
    d = d.loc[~duplicate_lat_lons]

    zestimate_null = pd.isnull(d[ses_col_to_use])
    print("%ss are null proportion %2.5f of time; removing" % (ses_col_to_use, zestimate_null.mean()))
    d = d.loc[~zestimate_null]
    assert pd.isnull(d['zillow_distance']).mean() == 0
    assert (d[ses_col_to_use] > 0).all()

    zestimate_out_of_range = d['zestimate'] > zestimate_truncation_threshold
    d['zestimate_prior_to_truncation_DO_NOT_USE'] = d['zestimate']
    print("Zestimates are out of range proportion %2.5f of the time; truncating" % zestimate_out_of_range.mean())
    d.loc[zestimate_out_of_range, 'zestimate'] = zestimate_truncation_threshold

    rent_zestimate_out_of_range = d['rent_zestimate'] > rent_zestimate_truncation_threshold
    d['rent_zestimate_prior_to_truncation_DO_NOT_USE'] = d['rent_zestimate']
    print("Rent zestimates are out of range proportion %2.5f of the time; truncating" % rent_zestimate_out_of_range.mean())
    d.loc[rent_zestimate_out_of_range, 'rent_zestimate'] = rent_zestimate_truncation_threshold

    geo_fields_we_need = ['new_census_p_black', 
                          'new_census_p_white', 
                          'new_census_mean_household_income', 
                          'new_census_median_household_income',
                          'new_census_state_fips_code',
                          'new_census_county_fips_code', 
                          'new_census_GEOID_Data',
                          'new_census_TRACTCE',
                          'census_tract', 
                          'city', 
                          'addr_zip', 
                          'addr_zip_first_5']
    for c in geo_fields_we_need:
        is_null = pd.isnull(d[c])
        print("Removing a small number of rows where %s is null: %i rows (proportion %2.5f)" % (c, is_null.sum(), is_null.mean()))
        d = d.loc[~is_null]

    zillow_match_too_far_away = d['zillow_distance'] > zillow_distance_threshold_in_meters
    print("Fraction of zillow matches further than %2.3f meters: %2.5f; filtering out" % (zillow_distance_threshold_in_meters, zillow_match_too_far_away.mean()))
    d = d.loc[~zillow_match_too_far_away]
    assert pd.isnull(d['zillow_lat']).sum() == 0
    assert pd.isnull(d['zillow_lon']).sum() == 0
    corelogic_match_too_far_away = d['dist_cl'] > corelogic_distance_threshold_in_meters
    print("Fraction of CoreLogic matches further than %2.3f meters: %2.5f; filtering out" % (corelogic_distance_threshold_in_meters, corelogic_match_too_far_away.mean()))
    d = d.loc[~corelogic_match_too_far_away]

    assert d['state'].map(lambda x:x in JUST_50_STATES_PLUS_DC).all()
    
    print("Analyzing many people mapped to exactly the same address")
    counts_by_location = d.groupby(['full_addr', 'addr_zip_first_5', 'state']).size().reset_index()
    counts_by_location.columns = ['full_addr', 'addr_zip_first_5', 'state', 'n_mapped_to_this_location']
    for cutoff in range(1, 100):
        print('Fraction %2.3f of addresses have count greater than %i' % 
              (counts_by_location.loc[counts_by_location['n_mapped_to_this_location'] > cutoff, 'n_mapped_to_this_location'].sum() / counts_by_location['n_mapped_to_this_location'].sum(), cutoff))
    old_len = len(d)
    d = pd.merge(d, counts_by_location, on=['full_addr', 'addr_zip_first_5', 'state'], how='left')
    assert len(d) == old_len
    assert pd.isnull(d['n_mapped_to_this_location']).sum() == 0

    if max_count_by_location_for_single_family_residence is not None:
        bad_idxs = (d['n_mapped_to_this_location'] > max_count_by_location_for_single_family_residence) & (d['use_code'] == 'SingleFamily')
        print("Removing fraction %2.3f which have too many people at the same location for a single family residence" % (bad_idxs.mean()))
        d = d.loc[~bad_idxs]
    d.index = range(len(d))

    print("Final number of rows returned: %i" % len(d))
    return d

def load_home_locations_for_prefixes(prefixes, verbose=True, usecols=None, filter_users_with_high_fraction_of_duplicate_pings=True, add_other_msa=True, dask_kwargs=None):
    if dask_kwargs is None:
        dask_kwargs = {}
    print("loading home locations for %i prefixes" % len(prefixes))
    filenames = [home_locations_for_id_prefix_path(prefix) for prefix in prefixes]
    d = load_csv_possibly_with_dask(filenames, use_dask=True, usecols=usecols, **dask_kwargs)
    if verbose:
        print("After reading in all home locations for prefixes, %i rows" % len(d))
    if filter_users_with_high_fraction_of_duplicate_pings:
        frac_duplicate = pd.read_csv(FRACTION_OF_DUPLICATE_PINGS_PER_USER_PATH)
        safegraph_ids_to_filter = set(frac_duplicate[frac_duplicate['frac_duplicate'] >
            DUPLICATE_PINGS_BLACKLIST_FRACTION_THRESHOLD]['safegraph_id'])
        d = d[~d['safegraph_id'].isin(safegraph_ids_to_filter)]
        if verbose:
            print("After filtering for users with over %f fraction of duplicate pings, %i rows" % (
                DUPLICATE_PINGS_BLACKLIST_FRACTION_THRESHOLD, len(d)))
    if add_other_msa and 'new_census_CBSA Title' in d.columns:
        assert not (d['new_census_CBSA Title'] == 'Other').any()
        assert d.loc[d['new_census_CBSA Title'].isna()]['new_census_Metropolitan/Micropolitan Statistical Area'].isna().all()
        d = d.fillna({'new_census_CBSA Title': 'Other'})
        assert not d['new_census_CBSA Title'].isna().any()
    d.index = range(len(d))
    return d

def load_all_home_locations(**kwargs):
    return load_home_locations_for_prefixes(VALID_ID_PREFIXES, **kwargs)

def compute_users_whose_paths_cross():
    """
    Compute all users whose paths cross.
    """
    all_users_with_home_locations = set(load_all_home_locations(usecols=['safegraph_id'])['safegraph_id'])
    for utc_day, utc_hour in itertools.product(UTC_DAYS_TO_USE_IN_ANALYSIS, VALID_UTC_HOURS):
        outfile_name = get_UNFILTERED_path_crossings_outfile(utc_day=utc_day, utc_hour=utc_hour)
        os.makedirs(os.path.dirname(outfile_name), exist_ok=True)
        outfile = open_file_for_exclusive_writing(outfile_name)
        if outfile is None: continue # No need to run job.
        with outfile: # The 'with' will close the file when done.
            do_compute_users_whose_paths_cross(utc_day, utc_hour, all_users_with_home_locations, outfile, outfile_name)

def do_compute_users_whose_paths_cross(utc_day, utc_hour, all_users_with_home_locations, outfile, outfile_name):
    """
    Compute all users whose paths cross in a given UTC day and hour, and dump to an outfile.
    """

    t_start = time.time()

    separation_in_seconds = 300
    separation_in_meters = MAX_PATH_CROSSING_THRESHOLD_TO_USE_IN_INITIAL_FILTERING # we can filter down further if we want because we store distances. 

    all_files = list_files_in_range('locations', 
                                    min_utc_hour=utc_hour,
                                    max_utc_hour=utc_hour,
                                    utc_days=[utc_day])

    current_datetime = datetime.datetime.strptime(utc_day + ' ' + utc_hour, '%Y_%m_%d %H')
    previous_datetime = current_datetime - datetime.timedelta(hours=1)
    previous_utc_day = previous_datetime.strftime('%Y_%m_%d')
    previous_utc_hour = previous_datetime.strftime('%H')
    min_timestamp = current_datetime.replace(tzinfo=pytz.utc).timestamp() - separation_in_seconds # used for filtering out rows in the preceding dataframe which are not near the boundary. 


    if previous_utc_day not in UTC_DAYS_TO_USE_IN_ANALYSIS:
        all_preceding_files = []
    else:
        all_preceding_files = list_files_in_range('locations', 
                                                  min_utc_hour=previous_utc_hour, 
                                                  max_utc_hour=previous_utc_hour,
                                                  utc_days=[previous_utc_day])
        assert len(all_preceding_files) == len(all_files)

    print("Number of files to read: %i" % len(all_files))
    all_dataframes = []
    for file_idx, filename in enumerate(all_files):
        cols_to_keep = ['safegraph_id', 'latitude', 'longitude', 'utc_timestamp', 'geo_hash', 'local_datetime', 'horizontal_accuracy']
        
        d_for_id_prefix = load_ping_file(file_path=filename, usecols=cols_to_keep)
        all_dataframes.append(d_for_id_prefix)
        print("Reading file %i/%i; %i rows added" % (file_idx+1, len(all_files), len(d_for_id_prefix)))

        if len(all_preceding_files) > 0:
            preceding_d = load_ping_file(file_path=all_preceding_files[file_idx], usecols=cols_to_keep)
            preceding_d = preceding_d.loc[preceding_d['utc_timestamp'] >= min_timestamp]
            preceding_d.index = range(len(preceding_d))
            print("%i rows added from preceding dataframe" % len(preceding_d))
            all_dataframes.append(preceding_d)

    d = pd.concat(all_dataframes)

    print("prior to filtering for %i users with home locations, %i rows" % (len(all_users_with_home_locations), len(d)))
    d = d.loc[d['safegraph_id'].isin(all_users_with_home_locations)].copy()
    print("After filtering for %i users with home locations, %i rows" % (len(all_users_with_home_locations), len(d)))

    print("Number of rows: %i" % len(d))
    all_path_crossings = kd_tree_pairs_of_users_whose_paths_cross(d, 
                                                                  separation_in_seconds=separation_in_seconds, 
                                                                  separation_in_meters=separation_in_meters,
                                                                  utc_day=utc_day,
                                                                  utc_hour=utc_hour)

    print("writing outfile to %s" % outfile_name)
    temp_outfile_name = outfile_name + '.tmp'
    all_path_crossings.to_csv(temp_outfile_name, compression='gzip')
    shutil.copyfileobj(open(temp_outfile_name, 'rb'), outfile)
    os.unlink(temp_outfile_name)
    print("Successfully computed path crossings for UTC day %s and UTC hour %s in %2.3f seconds" % (utc_day, utc_hour, time.time() - t_start))

def kd_tree_pairs_of_users_whose_paths_cross(d, separation_in_seconds, separation_in_meters, utc_day, utc_hour):
    """
    Given a dataframe of timestamped locations for users, return a set of pairs of users (ie, pairs of safegraph ids) 
    who are within separation_in_meters meters of each other within separation_in_seconds.
    
    Use a KD tree to compute all pairs that cross.
    Seems 4x faster than previous method creating one kd tree for each geohash.
    All this is done over narrow timeslices (because we shouldn't be comparing rows that are like an hour apart). 
    """

    current_start_time = d['utc_timestamp'].min()
    max_time = d['utc_timestamp'].max()

    d = d.sort_values(by=['safegraph_id'])
        
    timeslice_width = separation_in_seconds * 2
    all_pairs = []
    pairs_found = 0
    while (current_start_time <= max_time + 60): 
        start_of_computation_time_for_timeslice = time.time() # keep track of how long all this takes to do.
        current_end_time = min(max_time + 60, current_start_time + timeslice_width)

        print("\n\n\n*****Computing pairs for timeslice %s - %s" % 
              (datetime_from_utc_timestamp(current_start_time), 
               datetime_from_utc_timestamp(current_end_time)))
        time_slice_idxs = (d['utc_timestamp'] >= current_start_time) & (d['utc_timestamp'] < current_end_time)
        d_for_timeslice = copy.deepcopy(d.loc[time_slice_idxs]) # avoid modifying original dataframe. 
        d_for_timeslice.index = range(len(d_for_timeslice))
        current_start_time += separation_in_seconds # increment time slice. 

        print('Number of rows: %i' % len(d_for_timeslice))
        if len(d_for_timeslice) == 0:
            continue

        xs, ys, zs = lonlat_to_xyz(lon=d_for_timeslice['longitude'].values, 
            lat=d_for_timeslice['latitude'].values)

        print("Computing KD tree")
        t_start = time.time() # monitor how long this takes to do. 
        spatial_locs = np.array([xs, ys, zs]).transpose()
        kd_tree = cKDTree(spatial_locs)
        print("Seconds to compute tree: %2.3f" % (time.time() - t_start))

        t_start = time.time()
        pairs_indices = kd_tree.query_pairs(separation_in_meters, output_type='ndarray')

        pairs_ts_diff = np.abs(d_for_timeslice['utc_timestamp'].iloc[pairs_indices[:,0]].values -
            d_for_timeslice['utc_timestamp'].iloc[pairs_indices[:,1]].values)
        pairs_indices = pairs_indices[pairs_ts_diff <= separation_in_seconds]

        pairs_diff_safegraph_id = (d_for_timeslice['safegraph_id'].iloc[pairs_indices[:,0]].values !=
            d_for_timeslice['safegraph_id'].iloc[pairs_indices[:,1]].values)
        pairs_indices = pairs_indices[pairs_diff_safegraph_id]

        pairs_dists = np.linalg.norm(
            spatial_locs[pairs_indices[:,0]] - spatial_locs[pairs_indices[:,1]],
            axis=1)
        pairs_dataframe = {
            'dist': pairs_dists,
        }
        desired_fields = ['safegraph_id', 'local_datetime', 'utc_timestamp', 'latitude', 'longitude', 'geo_hash', 'horizontal_accuracy']
        for field in desired_fields:
            pairs_dataframe['a_'+field] = d_for_timeslice[field].iloc[pairs_indices[:,0]].values
            pairs_dataframe['b_'+field] = d_for_timeslice[field].iloc[pairs_indices[:,1]].values
        pairs_dataframe = pd.DataFrame(pairs_dataframe)
        pairs_found += len(pairs_dataframe)
        all_pairs.append(pairs_dataframe)
        print("Seconds to compute neighbors using tree: %2.3f" % (time.time() - t_start))
        print("Total time to process timeslice: %2.3f seconds" % (time.time() - start_of_computation_time_for_timeslice))
        print("Total number of path crossings identified after analyzing this timeslice (including all previous timeslices): %i" % pairs_found)
    all_pairs_whose_paths_cross = pd.concat(all_pairs)
    print("Prior to dropping duplicates, %i rows" % len(all_pairs_whose_paths_cross))
    all_pairs_whose_paths_cross = all_pairs_whose_paths_cross.drop_duplicates() # because timeslices overlap, have to deduplicate
    print("Identified %i path crossings after de-duplicating" % len(all_pairs_whose_paths_cross))

    all_pairs_of_users_whose_paths_cross = consolidate_path_crossings(
        all_pairs_whose_paths_cross, utc_day, utc_hour)
    return all_pairs_of_users_whose_paths_cross

def consolidate_path_crossings(all_pairs_whose_paths_cross, utc_day, utc_hour):
    """
    all_pairs_whose_paths_cross may have multiple rows for each user pair. 
    Here we combine all these rows into one row for every PATH_CROSSING_CONSOLIDATION_INTERVAL seconds.
    """
    print("Condensing path crossings dataframe into a dataframe with one row for each pair of users every %d seconds" %
        PATH_CROSSING_CONSOLIDATION_INTERVAL)
    assert (all_pairs_whose_paths_cross['a_safegraph_id'] < all_pairs_whose_paths_cross['b_safegraph_id']).all()
    assert len(all_pairs_whose_paths_cross.drop_duplicates()) == len(all_pairs_whose_paths_cross)

    t_start = time.time()

    crossing_timestamps = pd.concat([all_pairs_whose_paths_cross['a_utc_timestamp'], all_pairs_whose_paths_cross['b_utc_timestamp']], axis=1).min(axis=1)
    assert crossing_timestamps.isna().sum() == 0
    all_pairs_whose_paths_cross = all_pairs_whose_paths_cross.assign(
        utc_timestamp=crossing_timestamps,
        consolidation_key=crossing_timestamps.floordiv(PATH_CROSSING_CONSOLIDATION_INTERVAL),
    )

    grouped_d = all_pairs_whose_paths_cross.groupby(['a_safegraph_id', 'b_safegraph_id', 'consolidation_key'])
    one_row_d = []
    total_groups = len(grouped_d)
    n = 0
    for group_id, small_d in grouped_d:
        if n % 10000 == 0:
            print('%i/%i consolidated path crossings processed' % (n, total_groups))
        n += 1
        a_safegraph_id, b_safegraph_id, consolidation_key = group_id
        min_utc_timestamp = min(small_d['a_utc_timestamp'].min(), small_d['b_utc_timestamp'].min())
        max_utc_timestamp = max(small_d['a_utc_timestamp'].max(), small_d['b_utc_timestamp'].max())
        assert small_d['dist'].min() < MAX_PATH_CROSSING_THRESHOLD_TO_USE_IN_INITIAL_FILTERING

        ts_diff = (small_d['a_utc_timestamp'] - small_d['b_utc_timestamp']).abs()

        one_row = {
            'a_safegraph_id': a_safegraph_id, 
            'b_safegraph_id': b_safegraph_id, 
            'n_path_crossings': len(small_d),
            'a_pings':len(set(small_d['a_utc_timestamp'])), # not all rows correspond to unique pings for each person because one ping for one person may match to multiple pings for the other. 
            'b_pings':len(set(small_d['b_utc_timestamp'])),
            'a_min_local_datetime':small_d['a_local_datetime'].min(),
            'a_max_local_datetime':small_d['a_local_datetime'].max(), 
            'b_min_local_datetime':small_d['b_local_datetime'].min(),
            'b_max_local_datetime':small_d['b_local_datetime'].max(),
            'min_utc_timestamp':min_utc_timestamp, 
            'max_utc_timestamp':max_utc_timestamp, 
            'min_ts_diff':ts_diff.min(),
            'max_ts_diff':ts_diff.max(),
            'min_dist':small_d['dist'].min(), 
            'median_dist':np.median(small_d['dist']), 
            'mean_dist':small_d['dist'].mean(),
            'max_dist':small_d['dist'].max(), 
            'min_latitude':min(small_d['a_latitude'].min(), small_d['b_latitude'].min()), 
            'min_longitude':min(small_d['a_longitude'].min(), small_d['b_longitude'].min()), 
            'max_latitude':max(small_d['a_latitude'].max(), small_d['b_latitude'].max()),
            'max_longitude':max(small_d['a_longitude'].max(), small_d['b_longitude'].max()), 
            'median_latitude':np.median(list(set(small_d['a_latitude'])) + list(set(small_d['b_latitude']))), 
            'median_longitude':np.median(list(set(small_d['a_longitude'])) + list(set(small_d['b_longitude']))),
            'min_horizontal_accuracy':min(small_d['a_horizontal_accuracy'].min(), small_d['b_horizontal_accuracy'].min()),
            'max_horizontal_accuracy':max(small_d['a_horizontal_accuracy'].max(), small_d['b_horizontal_accuracy'].max()),
        }
        for t in PATH_CROSSING_ADDITIONAL_THRESHOLDS:
            k = f'satisfies_{t.distance_meters}m_{t.time_seconds}s'
            one_row[k] = (
                (small_d['dist'] <= t.distance_meters) &
                ((small_d['a_utc_timestamp'] - small_d['b_utc_timestamp']).abs() <= t.time_seconds)
            ).any().astype('int32')
        one_row_d.append(one_row)
    one_row_d = pd.DataFrame(one_row_d)
    print("Seconds to condense path crossings dataframe: %2.3f" % (time.time() - t_start))
    return one_row_d

def get_UNFILTERED_path_crossings_outfile(utc_day, utc_hour, annotated=False):
    base_dir = ANNOTATED_PATH_CROSSINGS_DIR if annotated else PATH_CROSSINGS_DIR
    return os.path.join(base_dir, 'path_crossings_utc_day_%s_utc_hour_%s.csv.gz' % (utc_day, utc_hour))

def get_number_of_pings_per_user_outfile(utc_day, utc_hour):
    """
    small helper method: return the number of total pings for each user
    """
    return os.path.join(PATH_CROSSINGS_DIR, 'pings_per_user_utc_day_%s_utc_hour_%s.csv.gz' % (utc_day, utc_hour))

def get_duplicate_pings_outfile(utc_day, utc_hour):
    """
    small helper method: return identical pings in 2+ users

    Note: Emma accidentally deleted this on master, putting it back in now, no need to check it again.
    """
    return os.path.join(PATH_CROSSINGS_DIR, 'duplicate_pings_utc_day_%s_utc_hour_%s.csv.gz' % (utc_day, utc_hour))

def load_number_of_pings_per_user(utc_day_prefix='', max_to_load=None):
    """
    Read in number of pings per user files.
    """
    t_start = time.time()
    utc_days_hours = [(utc_day, utc_hour)
        for utc_day, utc_hour in itertools.product(UTC_DAYS_TO_USE_IN_ANALYSIS, VALID_UTC_HOURS)
        if utc_day.startswith(utc_day_prefix)]
    if max_to_load is not None:
        utc_days_hours = utc_days_hours[:max_to_load]
    print("%i files to load for number of pings per user" % len(utc_days_hours))

    all_number_of_pings_per_user = Counter()
    total_rows_loaded = 0
    for utc_day, utc_hour in tqdm_wrap(utc_days_hours):
        f = get_number_of_pings_per_user_outfile(utc_day=utc_day, utc_hour=utc_hour)
        d = pd.read_csv(f)
        total_rows_loaded += len(d)
        for safegraph_id, num_pings in zip(d['safegraph_id'], d['num_pings']):
            all_number_of_pings_per_user[safegraph_id] += num_pings

    print("%i distinct users in %i rows" % (len(all_number_of_pings_per_user), total_rows_loaded))
    print("Seconds for loading number of pings per user: %2.3f" % (time.time() - t_start))
    return pd.Series(all_number_of_pings_per_user)

def run_all_jobs_for_compute_duplicate_pings():
    all_users_with_home_locations = set(load_all_home_locations(usecols=['safegraph_id'], filter_users_with_high_fraction_of_duplicate_pings=False)['safegraph_id'])
    for utc_day, utc_hour in itertools.product(UTC_DAYS_TO_USE_IN_ANALYSIS, VALID_UTC_HOURS):
        outfile_name = get_duplicate_pings_outfile(utc_day=utc_day, utc_hour=utc_hour)
        os.makedirs(os.path.dirname(outfile_name), exist_ok=True)
        outfile = open_file_for_exclusive_writing(outfile_name)
        if outfile is None: continue # No need to run job.
        with outfile: # The 'with' will close the file when done.
            compute_duplicate_pings(utc_day, utc_hour, outfile, outfile_name, all_users_with_home_locations)

def compute_duplicate_pings(utc_day, utc_hour, outfile, outfile_name, all_users_with_home_locations):
    """
    Compute identical pings in 2+ users in a given UTC day and hour, and dump to an outfile.
    """
    t_start = time.time()

    all_files = list_files_in_range('locations',
                                    min_utc_hour=utc_hour,
                                    max_utc_hour=utc_hour,
                                    utc_days=[utc_day])
    assert len(all_files) == 256
    d = []
    for file_idx, filename in enumerate(all_files):
        d_for_id_prefix = load_ping_file(file_path=filename,
            usecols=['safegraph_id', 'latitude', 'longitude', 'utc_timestamp', 'geo_hash', 'local_datetime', 'horizontal_accuracy'])
        d.append(d_for_id_prefix)
        print("Reading file %i/%i; %i rows added" % (file_idx+1, len(all_files), len(d_for_id_prefix)))

    d = pd.concat(d)
    print("Number of rows: %i" % len(d))

    print("prior to filtering for %i users with home locations, %i rows" % (
        len(all_users_with_home_locations), len(d)))
    d = d.loc[d['safegraph_id'].map(lambda x:x in all_users_with_home_locations)].copy()
    print("After filtering for %i users with home locations, %i rows" % (
        len(all_users_with_home_locations), len(d)))

    d_counts = d['safegraph_id'].value_counts()
    d_counts.name = 'num_pings'
    print("Number of distinct users: %i" % len(d_counts))
    d_counts.to_csv(get_number_of_pings_per_user_outfile(utc_day, utc_hour), header=True,
        index_label='safegraph_id', compression='gzip')

    num_total_pings = len(d)

    print("Finding duplicate pings")
    assert d.duplicated().sum() == 0 # don't need this, but a good sanity check.
    d = d.loc[d[['latitude', 'longitude', 'utc_timestamp']].duplicated(keep=False)]
    num_duplicate_pings = len(d)
    print("Found %d/%d (%.2f%%) duplicate pings" % (
        num_duplicate_pings, num_total_pings, num_duplicate_pings/num_total_pings*100))

    print("Writing outfile to %s" % outfile_name)
    temp_outfile_name = outfile_name + '.tmp'
    d.to_csv(temp_outfile_name, compression='gzip')
    shutil.copyfileobj(open(temp_outfile_name, 'rb'), outfile)
    os.unlink(temp_outfile_name)
    print("Successfully computed duplicate pings for UTC day %s and UTC hour %s "
          "in %2.3f seconds" % (utc_day, utc_hour, time.time() - t_start))

def load_all_duplicate_pings(use_dask=True):
    """
    Loads duplicate pings for all days and hours.
    """
    duplicate_pings_filenames = [
        get_duplicate_pings_outfile(utc_day=utc_day, utc_hour=utc_hour)
        for utc_day in UTC_DAYS_TO_USE_IN_ANALYSIS
        for utc_hour in VALID_UTC_HOURS]
    return load_csv_possibly_with_dask(duplicate_pings_filenames, use_dask=use_dask)

def compute_fraction_of_duplicate_pings_per_user():
    """
    Computes a mapping from safegraph ID to fraction of duplicate pings, and saves it to a file.
    Takes a few hours (mainly for load_number_of_pings_per_user).
    """
    t_start = time.time()
    print('Loading number of pings per user')
    all_number_of_pings_per_user = load_number_of_pings_per_user()
    print('Loading duplicate pings')
    all_duplicate_pings = load_all_duplicate_pings()
    print('Computing fraction of duplicate pings per user')
    duplicate_pings_by_safegraph_id = all_duplicate_pings['safegraph_id'].value_counts()
    total_pings = all_number_of_pings_per_user[duplicate_pings_by_safegraph_id.index]
    frac_duplicate = duplicate_pings_by_safegraph_id / total_pings
    frac_duplicate.name = 'frac_duplicate'
    frac_duplicate.to_csv(FRACTION_OF_DUPLICATE_PINGS_PER_USER_PATH, header=True,
        index_label='safegraph_id', compression='gzip')
    print("Successfully computed fraction of duplicate pings per user in %2.3f seconds" % (
        time.time() - t_start))

def get_msa_census_info():
    """
    Returns a dict with the following:
    - micro_areas: Set of names of micropolitan statistical areas
    - metro_areas: Set of names of metropolitan statistical areas
    - population:  Census MSA population in 2017
    """
    filename = os.path.join(BASE_DIRECTORY_FOR_EVERYTHING, 'cbsa-est2018-alldata.csv')
    cbsa_data = pd.read_csv(filename, encoding='windows-1252')
    cbsa_data = cbsa_data.drop_duplicates(subset='NAME')
    return {
        'micro_areas': set(cbsa_data[cbsa_data['LSAD']=='Micropolitan Statistical Area']['NAME']),
        'metro_areas': set(cbsa_data[cbsa_data['LSAD']=='Metropolitan Statistical Area']['NAME']),
        'population': cbsa_data[['NAME', 'POPESTIMATE2017']].set_index('NAME')['POPESTIMATE2017'],
    }

def load_path_crossing_segregation_file(filename, metro_or_micro, msa_census_info):
    """
    Loads path crossing segregation file as DataFrame.
    Default settings should be used in the paper.
    - filename:        Filename of pickle file
    - metro_or_micro:  'metro' or 'micro' to filter metropolitan or micropolitan statistical areas
    - msa_census_info: from get_msa_census_info()
    """
    path = os.path.join(ECONOMIC_SEGREGATION_MEASURES_DIR, filename)
    results = pickle.load(open(path, 'rb'))
    results_key = [x for x in results if x.startswith('path_crossings_with_mixed_model+')][0]
    results = results[results_key]
    results = results[~results['segregation_index'].isna()]
    results = results[results['group'].isin(msa_census_info[metro_or_micro+'_areas'])]
    return results

class HomeCensusAreaMapper:
    """
    Determine whether path crossings occur in either person's home Census tract/Census Block Group (the latter is smaller). 
    Relies on home locations and Census shapefiles. 

    If initialized with its default arguments (which is what you want to use in general) this class will take a while to be initialized. 
    You can test it quickly by just loading data from Virginia and only loading a few home locations. 

    x = HomeCensusAreaMapper('ACS_2017_5YR_BG_51_VIRGINIA.gdb', 16) #

    able_to_map, features_mapped_to = x.query(lat=path_crossings['median_latitude'].values, 
            lon=path_crossings['median_longitude'].values,
            a_safegraph_ids=path_crossings['a_safegraph_id'].values,
            b_safegraph_ids=path_crossings['b_safegraph_id'].values)
    
    """
    def __init__(self, gdb_filename='ACS_2017_5YR_BG.gdb', n_id_prefixes_to_load=256):
        print("Loading Census data from %s" % gdb_filename)
        self.gdb_filename = gdb_filename
        self.n_id_prefixes_to_load = n_id_prefixes_to_load
        self.cbg_data = CensusBlockGroups(gdb_files=[gdb_filename])
        self.home_census_info = load_home_locations_for_prefixes(VALID_ID_PREFIXES[:n_id_prefixes_to_load], 
                                                                usecols=['safegraph_id', 
                                                                         'new_census_state_fips_code', 
                                                                         'new_census_county_fips_code', 
                                                                         'new_census_GEOID_Data', 
                                                                         'new_census_TRACTCE'])
    def query(self, lat, lon, a_safegraph_ids, b_safegraph_ids):
        path_crossing_census_info = self.cbg_data.get_demographic_stats_of_points(latitudes=lat, 
                                    longitudes=lon, 
                                    desired_cols=[])[['state_fips_code', 'county_fips_code', 'TRACTCE', 'GEOID_Data']]
        path_crossing_census_info.columns = ['path_crossing_%s' % a for a in path_crossing_census_info.columns]
        path_crossing_census_info['a_safegraph_id'] = a_safegraph_ids
        path_crossing_census_info['b_safegraph_id'] = b_safegraph_ids
        
        a_home_census_info = self.home_census_info.copy()
        a_home_census_info.columns = ['a_' + a for a in a_home_census_info.columns]
        b_home_census_info = self.home_census_info.copy()
        b_home_census_info.columns = ['b_' + a for a in b_home_census_info.columns]
        combined_census_info = (path_crossing_census_info
                        .merge(a_home_census_info,
                               how='left',
                               on='a_safegraph_id',
                               validate='many_to_one')
                        .merge(b_home_census_info,
                               how='left',
                               on='b_safegraph_id', 
                               validate='many_to_one'))
        assert len(combined_census_info) == len(lat) # make sure length hasn't changed. 
        
        cols_that_must_be_equal = {'census_tract':['state_fips_code', 'county_fips_code', 'TRACTCE'], 
                                   'census_block_group':['state_fips_code', 'county_fips_code', 'GEOID_Data']}

        for area in cols_that_must_be_equal:
            for person in ['a', 'b']:
                path_crossing_in_census_area = np.array([True for a in range(len(combined_census_info))])
                for col in cols_that_must_be_equal[area]:
                    person_home_col = '%s_new_census_%s' % (person, col)
                    path_crossing_col = 'path_crossing_%s' % col
                    if col != 'GEOID_Data': # avoid weird hanging zeros by casting everything to a float. 
                        combined_census_info[person_home_col] = combined_census_info[person_home_col].astype(float)
                        combined_census_info[path_crossing_col] = combined_census_info[path_crossing_col].astype(float)
                    fields_match = combined_census_info[person_home_col] == combined_census_info[path_crossing_col]
                    assert pd.isnull(fields_match).sum() == 0
                    col_1_has_values = (~pd.isnull(combined_census_info[path_crossing_col])).mean()
                    col_2_has_values = (~pd.isnull(combined_census_info[person_home_col])).mean()
                    print("Fraction of column %s with values: %2.3f" % (path_crossing_col, col_1_has_values.mean()))
                    print("Fraction of column %s with values: %2.3f" % (person_home_col, col_2_has_values.mean()))
                    assert combined_census_info[person_home_col].dtype == combined_census_info[path_crossing_col].dtype
                    if self.gdb_filename == 'ACS_2017_5YR_BG.gdb':
                        assert col_1_has_values > .999
                        assert col_2_has_values > .999
                    path_crossing_in_census_area = path_crossing_in_census_area & fields_match.values
                combined_census_info['in_%s_%s' % (person, area)] = path_crossing_in_census_area
                print('%i/%i path crossings occur in %s home %s' % (
                    path_crossing_in_census_area.sum(), 
                    len(path_crossing_in_census_area),
                    person, 
                    area))
                
            combined_census_info['in_same_%s' % area] =  np.logical_and(combined_census_info['in_a_%s' % area].values,
                                                                        combined_census_info['in_b_%s' % area].values)
            combined_census_info['visiting_%s' % area] = np.logical_xor(combined_census_info['in_a_%s' % area].values,
                                                                        combined_census_info['in_b_%s' % area].values)
            combined_census_info['outside_%s' % area] = np.logical_not(
                np.logical_or(combined_census_info['in_a_%s' % area].values,
                              combined_census_info['in_b_%s' % area].values))

        assert combined_census_info.loc[combined_census_info['in_a_census_block_group'].values == True, 'in_a_census_tract'].all()
        assert combined_census_info.loc[combined_census_info['in_b_census_block_group'].values == True, 'in_b_census_tract'].all()

        able_to_map = []
        features_mapped_to = []
        cols_of_interest = ['in_same_census_tract', 'visiting_census_tract', 'outside_census_tract',
                            'in_same_census_block_group', 'visiting_census_block_group', 'outside_census_block_group']
        boolean_vals = combined_census_info[cols_of_interest].values #cast to numpy array for speed. 
        for i in range(len(combined_census_info)):
            row = boolean_vals[i, :]
            able_to_map.append(row.sum() > 0)
            if row.sum() > 0:
                features_for_row = []
                for j in range(len(row)):
                    if row[j]:
                        features_for_row.append(cols_of_interest[j])
                features_mapped_to.append(json.dumps(sorted(features_for_row)))
        able_to_map = np.array(able_to_map)
        features_mapped_to = np.array(features_mapped_to)
        print("Total mapping counts (out of %i lat,lons)" % len(lat))
        print(Counter([a for b in features_mapped_to for a in json.loads(b)]))
        return able_to_map, features_mapped_to


def get_safegraph_places_geometry_filenames():
    return sorted(
        os.path.join(SAFEGRAPH_PLACES_DIR, f)
        for f in os.listdir(SAFEGRAPH_PLACES_DIR)
        if f.startswith('core_poi-geometry-'))


class SafegraphPlacesMapper:
    """
    Maps lats/lons to Safegraph Places.

    Unlike other mappers, this mapper not only maps to features, but also individual POI identifiers.
    It outputs a list of elements, where each element is like
        ["722513", ["sg:0b7fadcb8c5549d1a9754e36c392de97", "sg:3a1740c0b2b34506896dea0ee6d1968a"]]
    722513 is a 6-digit NAICS code, and the two other strings are Safegraph Places ID.

    By mappping to fine-grained 6-digit NAICS codes, this gives us the freedom to write out
    coarser-grained files (e.g. 4-digit codes) when we write out stratified path crossings.
    """
    def __init__(self, max_num_files=None):
        filenames = get_safegraph_places_geometry_filenames()
        if max_num_files is not None and len(filenames) > max_num_files:
            filenames = filenames[:max_num_files]
        print('Loading %d files for safegraph places' % len(filenames))
        gdf = load_csv_possibly_with_dask(filenames,
            usecols=['safegraph_place_id', 'polygon_wkt', 'naics_code'],
            dtype='str')
        gdf['polygon_wkt'] = gdf['polygon_wkt'].apply(wkt.loads)
        self.d = geopandas.GeoDataFrame(gdf, crs=WGS_84_CRS).set_geometry('polygon_wkt')


    def query(self, lat, lon):
        """
        This has no max_dist argument: we only map a lat,lon to a feature if it's in the interior of the feature. 
        """
        latitudes_and_longitudes = pd.DataFrame({'latitude':lat, 'longitude':lon, 'placeholder':range(len(lat))})
        path_crossings_geodataframe = geopandas.GeoDataFrame(
            latitudes_and_longitudes,  
            geometry=geopandas.points_from_xy(latitudes_and_longitudes['longitude'].values, 
                                              latitudes_and_longitudes['latitude'].values), 
            crs=WGS_84_CRS)

        points_near_features = geopandas.sjoin(self.d, path_crossings_geodataframe, how='inner', op='contains')
        idxs_to_features = {}
        for placeholder, small_d in points_near_features.groupby('placeholder'):
            all_tags = [
                [tag, sorted(list(set(small_d2['safegraph_place_id'])))]
                for tag, small_d2 in small_d[['naics_code', 'safegraph_place_id']].groupby('naics_code')]
            all_tags.sort()
            idxs_to_features[placeholder] = json.dumps(all_tags)

        idxs_that_were_matched = sorted(idxs_to_features.keys())
        has_a_match = np.array([False for a in range(len(lat))])
        has_a_match[idxs_that_were_matched] = True
        features_matched_to = [idxs_to_features[a] for a in idxs_that_were_matched]
        print("Total number of lat,lons that were matched %i/%i, proportion %2.3f" % (len(idxs_that_were_matched), len(lat), has_a_match.mean()))
        return has_a_match, features_matched_to
        
class TigerMapper:
    """
    Map to roads or rails etc using the Census-curated TIGER database. 
    """
    def __init__(self, tiger_layer_to_use):
        t0 = time.time()
        self.tiger_layer_to_use = tiger_layer_to_use
        if tiger_layer_to_use == 'Rails':
            full_path = os.path.join(BASE_DIRECTORY_FOR_EVERYTHING, 'tlgdb_2017_a_us_rails.gdb/')
        elif tiger_layer_to_use == 'Roads':
            full_path = os.path.join(BASE_DIRECTORY_FOR_EVERYTHING, 'tlgdb_2017_a_us_roads.gdb/')
        else:
            raise Exception("%s is not a valid tiger layer" % tiger_layer_to_use)
        print("Loading tiger features")
        layer_list = fiona.listlayers(full_path)
        print(layer_list)
        self.crs_with_units_of_meters = CRS_FOR_DISTANCE_COMPUTATIONS_IN_GEOPANDAS_IN_NORTH_AMERICA # See note in constants_and_util.py. We need to use a CRS that can do distance computations in meters, and this one agrees very well with our other distance computations. 
        self.tiger_data = geopandas.read_file(full_path, layer=tiger_layer_to_use).to_crs(self.crs_with_units_of_meters)
        print("Number of tiger features loaded: %i in %2.3f seconds" % (len(self.tiger_data), time.time() - t0))
        
    def query(self, lat, lon, max_dist):
        """
        Compute whether each lat,lon is within max_dist of a TIGER feature. This is a little convoluted. 

        It does seem to work, though. I tested it on rails and generated 100 interpolated lat,lons getting increasingly far from a railroad; 
        they stopped getting mapped to the railroad about when we'd expect them to. 
        """
        t0 = time.time()
        latitudes_and_longitudes = pd.DataFrame({'latitude':lat, 'longitude':lon, 'latlon_index':range(len(lat))})
        path_crossings_geodataframe = geopandas.GeoDataFrame(
            latitudes_and_longitudes,  
            geometry=geopandas.points_from_xy(latitudes_and_longitudes['longitude'].values, 
                                              latitudes_and_longitudes['latitude'].values),
                        crs=WGS_84_CRS)
        
        path_crossings_geodataframe = path_crossings_geodataframe.to_crs(self.crs_with_units_of_meters).buffer(distance=max_dist)

        path_crossings_geodataframe = geopandas.GeoDataFrame(
            latitudes_and_longitudes, 
            geometry=path_crossings_geodataframe, 
            crs=self.crs_with_units_of_meters
        )

        points_near_features = geopandas.sjoin(
            self.tiger_data,
            path_crossings_geodataframe,
            how='inner',
            op='intersects')
        idxs_that_were_matched = list(set(list(points_near_features['latlon_index'])))
        has_a_match = np.array([False for a in range(len(lat))])
        has_a_match[idxs_that_were_matched] = True

        points_near_features['human_readable_tiger_features'] = points_near_features['MTFCC'].map(TIGER_FEATURE_CODES).values

        tiger_features = points_near_features.groupby(
            ['latitude', 'longitude', 'latlon_index'])['human_readable_tiger_features'].apply(
            lambda x: json.dumps(sorted(list(set(x))))
        ).reset_index()      
        
        result = pd.merge(latitudes_and_longitudes, tiger_features, how='left', on=['latitude', 'longitude', 'latlon_index'])
        assert(len(lat) == len(result)) 
        assert(sum(has_a_match) == result['human_readable_tiger_features'].count())
        tiger_features_list = result['human_readable_tiger_features'].dropna().tolist()
        print("Total number of lat,lons that were matched within %2.3f meters: %i/%i, %2.3f" % (max_dist, len(idxs_that_were_matched), len(lat), has_a_match.mean()))
        print("Total time for computation: %2.3f seconds" % (time.time() - t0))

        return has_a_match, tiger_features_list

class CombinedPOIMapper:
    """
    Integrate all data sources for mapping points into a single unified interace we can use to run queries. 
    Note this can be used for either pings or path crossings. 
    """
    def __init__(self, source_names, use_small_prototyping_datasets, distance_thresholds=None, filter_open_street_maps_categories=False):
        assert all([a in ALL_FEATURE_MAPPING_DATA_SOURCES for a in source_names])
        self.sources = {}
        if distance_thresholds is None:
            self.distance_thresholds = {'tiger_rails':20, 'tiger_roads':20}
        else:
            self.distance_thresholds = distance_thresholds

        for source in source_names:
            source_t0 = time.time()
            print("\n\n\n*******Loading data for source %s" % source)
            if source == 'tiger_roads':
                if use_small_prototyping_datasets:
                    raise Exception("This takes a long time to load")
                self.sources[source] = TigerMapper(tiger_layer_to_use='Roads')
            elif source == 'tiger_rails':
                self.sources[source] = TigerMapper(tiger_layer_to_use='Rails')
            elif source == 'safegraph_places':
                self.sources[source] = SafegraphPlacesMapper(max_num_files=2 if use_small_prototyping_datasets else None)
            elif source == 'home_census':
                if use_small_prototyping_datasets:
                    n_id_prefixes_to_load = 16
                    gdb_file = 'ACS_2017_5YR_BG_51_VIRGINIA.gdb'
                else:
                    n_id_prefixes_to_load = 256
                    gdb_file = 'ACS_2017_5YR_BG.gdb'
                self.sources[source] = HomeCensusAreaMapper(gdb_file, n_id_prefixes_to_load=n_id_prefixes_to_load)
            else:
                raise Exception("%s is not a valid data source" % source)
            print("Loaded %s in %2.3f seconds" % (source, time.time() - source_t0))

    def query(self, lat, lon, groups=None, test_outfile_name=None):
        """
        This returns a dataframe in the same order as lat,lon. 
        For each feature_type in self.sources, the dataframe will have a column filled with strings. The string will be 
        "[]" if nothing in the feature type matches, and a list otherwise. (The strings are produced by calling json.dumps on the list). 

        If groups is not None, also return a dataframe with the fraction of each group which can be mapped. 
        """

        print("Prior to deduplicating, %i latlons to map" % len(lat))

        result = pd.DataFrame({'latitude':lat, 'longitude':lon})
        deduplicated = result.drop_duplicates().copy() # for speed, only map each lat,lon once, then join back to the original dataframe.
        print("After deduplicating, %i rows to map" % len(deduplicated))
        
        if groups is not None:
            result['group'] = groups

        for source in self.sources:
            print("\n\nMapping using %s" % source)
            t0_source = time.time()
            if source not in ['user_work', 'user_home', 'home_census']:
                deduplicated['%s_feature' % source] = '[]' 
            if source in ['tiger_roads', 'tiger_rails']:
                able_to_map, features_matched_to = self.sources[source].query(
                    lat=deduplicated['latitude'].values, 
                    lon=deduplicated['longitude'].values, 
                    max_dist=self.distance_thresholds[source])
                deduplicated.loc[able_to_map, '{}_feature'.format(source)] = features_matched_to
                assert len(able_to_map) == len(deduplicated) # make sure nothing weird happened. 
            elif source == 'safegraph_places':
                able_to_map, features_matched_to = self.sources[source].query(
                    lat=deduplicated['latitude'].values, 
                    lon=deduplicated['longitude'].values)
                deduplicated.loc[able_to_map, 'safegraph_places_feature'] = features_matched_to
                assert len(able_to_map) == len(deduplicated) # make sure nothing weird happened. 
            elif source == 'home_census':
                result['%s_feature' % source] = '[]' 
                able_to_map, features_matched_to = self.sources[source].query(
                    lat=result['latitude'].values,
                    lon=result['longitude'].values,
                    a_safegraph_ids=result['a_safegraph_id'].values,
                    b_safegraph_ids=result['b_safegraph_id'].values
		)
                result.loc[able_to_map, 'home_census_feature'] = features_matched_to
                assert len(able_to_map) == len(result) # make sure nothing weird happened.
            else:
                raise Exception("Not a valid source")
            
            
            print("Total time to map with %s: %2.3f seconds" % (source, time.time() - t0_source))

        print("\n\nDone mapping. Results:")
        old_len = len(result)
        old_result = result.copy()
        result = result.merge(deduplicated, on=['latitude', 'longitude'], how='left', validate='many_to_one')

        assert len(result) == old_len
        assert np.allclose(result['latitude'], old_result['latitude']) # make sure order doesn't change. Documentation implies it shouldn't: "left: use only keys from left frame, similar to a SQL left outer join; preserve key order.".
        assert np.allclose(result['longitude'], old_result['longitude'])

        mapped_to_any_source = np.array([False for a in range(len(result))])

        test_outfile = []
        for source in self.sources:
            mapped_to_source = result['%s_feature' % source].map(lambda x:x != '[]')
            print("Fraction of lat,lons mapped to source %s: %2.3f" % (source, mapped_to_source.mean()))
            mapped_to_any_source = mapped_to_any_source | mapped_to_source

            if test_outfile_name is not None:
                print("Random sample of rows mapped to %s" % source)
                mapped_rows = result.loc[mapped_to_source, ['latitude', 'longitude', '%s_feature' % source]].copy()
                mapped_rows.columns = ['latitude', 'longitude', 'source']
                test_outfile.append(mapped_rows.iloc[random.sample(range(len(mapped_rows)), min(50, len(mapped_rows)))])

        result['any_feature'] = [json.dumps(['any']) if mapped_to_any_source[i] else json.dumps([]) for i in range(len(mapped_to_any_source))]
        print("Mapped to any source: %2.3f" % mapped_to_any_source.mean())
        assert len(result.dropna()) == len(result)

        if test_outfile_name is not None:
            unmapped_rows = result.loc[~mapped_to_any_source, ['latitude', 'longitude', 'any_feature']].copy()
            unmapped_rows.columns = ['latitude', 'longitude', 'source']
            unmapped_rows['source'] = 'not mapped to any feature'
            unmapped_rows = unmapped_rows.iloc[random.sample(range(len(unmapped_rows)), min(50, len(unmapped_rows)))]
            test_outfile.append(unmapped_rows)

            test_outfile = pd.concat(test_outfile)
            test_outfile.to_csv(test_outfile_name)
        if groups is not None:
            print("Stratifying by group")
            group_fracs = None
            for a in result.columns:
                if a.endswith('_feature'):
                    df_to_group = result[['group', a]].copy()
                    df_to_group['has_feature'] = df_to_group[a].map(lambda x:x != '[]')
                    grouped_d = df_to_group[['group', 'has_feature']].groupby('group').mean().reset_index()
                    grouped_d.columns = ['group', a]
                    if group_fracs is None:
                        group_fracs = grouped_d
                    else:
                        group_fracs = pd.merge(group_fracs, grouped_d, on='group', how='inner')
                    assert len(group_fracs) == len(set(result['group']))

            group_fracs.index = group_fracs['group'].values
            print(group_fracs)
            return result, group_fracs
        else:
            return result

if __name__ == '__main__':
    functions = [
        'extract_middle_of_night_locations',
        'create_final_zestimate_files',
        'compute_duplicate_pings',
        'compute_fraction_of_duplicate_pings_per_user',
        'compute_users_whose_paths_cross',
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument('function_to_run', choices=functions)
    initial_args = sys.argv[1:2]
    secondary_args = sys.argv[2:]
    init_args = parser.parse_args(initial_args) # only parse the first two arguments for now

    parser = argparse.ArgumentParser()
    if init_args.function_to_run == 'extract_middle_of_night_locations':
        run_all_jobs_for_extract_middle_of_night_locations()
    elif init_args.function_to_run == 'create_final_zestimate_files':
        create_final_zestimate_files()
    elif init_args.function_to_run == 'compute_duplicate_pings':
        run_all_jobs_for_compute_duplicate_pings()
    elif init_args.function_to_run == 'compute_fraction_of_duplicate_pings_per_user':
        compute_fraction_of_duplicate_pings_per_user()
    elif init_args.function_to_run == 'compute_users_whose_paths_cross':
        compute_users_whose_paths_cross()
    else:
        raise Exception("Not a valid task.")
