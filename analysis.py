from constants_and_util import *
import os.path
import random
import shutil
import json
import re
import matplotlib.pyplot as plt
import copy
from collections import Counter, defaultdict, namedtuple
from scipy.stats import pearsonr, spearmanr, linregress, logistic
import numpy as np
import datetime
import pandas as pd
import seaborn as sns
import dataprocessor
from IPython import embed
import numpy as np
import pickle
import multiprocessing
import argparse
import sys
import itertools
import random
import fiona
import geopandas

def estimate_path_crossing_segregation_with_mixed_model(x, y, max_people_to_fit_model_on, rescale_x=True, rescale_y=True, maxiter=None, initialize_params=True):
    """
    Method that does the main work of estimating the mixed model. 
    x is a list of user SESs in a given MSA. 
    y is a list of lists, of the same length of x, with the SESs of users people in x cross path with. 
    We normalize all entries in x and y by subtracting the mean of x and dividing by its std. 
    IMPORTANT: this method assumes that x is z-scored and will not yield the correct estimates otherwise. 
    We normalize y just to keep things on the same scale + hopefully improve convergence. 
    """
    assert len(x) == len(y)
    x = copy.deepcopy(x)
    y = copy.deepcopy(y)
    if len(x) > max_people_to_fit_model_on:
        print("Downsampling (original length %i; downsampled length %i)" % (len(x), max_people_to_fit_model_on))
        idxs = sorted(random.sample(range(len(x)), max_people_to_fit_model_on))
        x = [x[i] for i in idxs]
        y = [y[i] for i in idxs]
    print("Fitting mixed model. Rescaling x: %s; rescaling y: %s" % (rescale_x, rescale_y))

    n_points_per_person = [len(y_i) for y_i in y]
    assert (np.array(n_points_per_person) > 0).all()
    print("Mean number of points per person: %2.3f; median: %2.3f; total points %i; total people %i" % (
        np.mean(n_points_per_person), np.median(n_points_per_person), sum(n_points_per_person), len(y)))

    if len(y) < 10 or sum(n_points_per_person) < 100:
        print("This is too little data to fit the model on; returning nans.")
        return np.nan, len(y), sum(n_points_per_person), {}

    x = np.array(x)
    x_mu = x.mean()
    x_sigma = x.std(ddof=1)

    if rescale_x:
        x = (x - x_mu) / x_sigma
    if rescale_y:
        y = [list((np.array(y_i) - x_mu)/x_sigma) for y_i in y]

    df = {'person':[], 
          'ses':[], 
          'other_person_ses':[]}
    for i in range(len(x)):
        n = len(y[i])
        df['person'] += [i] * n
        df['ses'] += [x[i]] * n
        df['other_person_ses'] += y[i]
    df = pd.DataFrame(df)

    import analysis_r
    mixed_model_results = analysis_r.fit_mixed_model_for_path_crossing_segregation(df)

    print(mixed_model_results)

    n_people = len(x)
    n_obs = len(df)

    if 'error' in mixed_model_results:
        return np.nan, n_people, n_obs, mixed_model_results

    a = mixed_model_results['a']
    noise_estimate = np.sqrt(mixed_model_results['random_effects_covariance'])
    print("Estimated noise_1 (std): %2.3f" % noise_estimate)
    print("Estimated alpha: %2.3f" % a)
    print("Estimated intercept: %2.3f" % mixed_model_results['b'])
    """
    Now estimate cor(ax + b + e, x) = 
    cor(x, ax + e) = 
    [because cor(x, y) = cov(x, y) / sqrt(var(x) * var(y))]
    cov(x, ax + e) / sqrt(var(x) * var(ax + e)) = 
    cov(x, ax) / sqrt(1 * var(ax + e)) = 
    a / sqrt(a^2 + noise_estimate^2) = 
    sqrt(a^2 / (a^2 + noise_estimate^2)))
    Again, the assumption is that x has variance 1, and this will not yield the correct estimates otherwise. 
    """
    mixed_model_segregation_estimate = np.sqrt((a ** 2) / (a ** 2 + noise_estimate ** 2)) 

    return mixed_model_segregation_estimate, n_people, n_obs, mixed_model_results

def bin_local_hour(h):
    """
    Divide 24-hour time into 8 bins. This can maybe be written more succinctly but it's transparent. 
    """
    assert h >= 0
    assert h < 24
    if h < 3:
        return '00-03'
    elif h < 6:
        return '03-06'
    elif h < 9:
        return '06-09'
    elif h < 12:
        return '09-12'
    elif h < 15:
        return '12-15'
    elif h < 18:
        return '15-18'
    elif h < 21:
        return '18-21'
    elif h < 24:
        return '21-24'
    else:
        raise Exception("Not a valid value")

def write_out_stratified_path_crossings(variable_to_stratify_by, max_to_load=None):
    """
    Stratify by feature type or local hour and write out files for each subset: 
    this allows us to more rapidly compute segregation using just a subset of the data. 
    """
    t0 = time.time()
    feature_to_combined_features = defaultdict(list)
    for combined_feature, features in COMBINED_FEATURES:
        for feature in features:
            feature_to_combined_features['feature_type='+feature].append(combined_feature)

    usecols=['a_safegraph_id', 'b_safegraph_id', 'min_utc_timestamp', 'a_min_local_datetime', 'median_latitude', 'median_longitude', 'min_dist']
    if variable_to_stratify_by == 'feature_type':
        annotated = True
        usecols += ['any_feature'] + ['%s_feature' % a for a in ALL_FEATURE_MAPPING_DATA_SOURCES]
    else:
        annotated = False

    naics6_to_feature_name = (
        pd.read_csv(SAFEGRAPH_PLACES_NAICS6_MAPPING_FILE, dtype=str)
        .set_index('naics_code')['feature_name'].to_dict())
    naics6_to_feature_name = {k: v for k, v in naics6_to_feature_name.items()
        if any(k.startswith(p) for p in SAFEGRAPH_PLACES_NAICS_PREFIXES)}
    feature_name_to_naics6 = {v: k for k, v in naics6_to_feature_name.items()}

    home_locations = dataprocessor.load_all_home_locations()

    all_path_crossings = load_path_crossings(
        utc_day_prefix='', 
        max_to_load=max_to_load, 
        use_dask=True, 
        usecols=usecols,
        annotated=annotated)

    temporal_path_crossings = compute_temporal_path_crossings(all_path_crossings, home_locations)
    temporal_path_crossings.set_index(['a_safegraph_id', 'b_safegraph_id'], inplace=True)

    assert (all_path_crossings['a_safegraph_id'] < all_path_crossings['b_safegraph_id']).all()

    if variable_to_stratify_by == 'feature_type':
        mapped_to_any_feature = pd.Series(dtype=bool, index=all_path_crossings.index)
        crossed_paths_not_on_road = pd.Series(dtype=bool, index=all_path_crossings.index)
        crossed_paths_not_on_primary_or_secondary_road = pd.Series(dtype=bool, index=all_path_crossings.index)

        combined_features = {combined_feature: pd.Series(dtype=bool, index=all_path_crossings.index)
                             for combined_feature, features in COMBINED_FEATURES}

        stratification_idxs = {}
        for feature_type in ['%s_feature' % a for a in ALL_FEATURE_MAPPING_DATA_SOURCES]:
            if feature_type == 'safegraph_places_feature':
                load_func = lambda x: [
                    naics6_to_feature_name[a[0]]
                    for a in json.loads(x)
                    if a[0] in naics6_to_feature_name]
            else:
                load_func = lambda x: json.loads(x)
            feature_sets = all_path_crossings[feature_type].map(load_func)
            val_counts = Counter([a for b in feature_sets for a in b])
            print(val_counts)

            if feature_type != 'home_census_feature':
                is_mapped = feature_sets.map(lambda x: len(x) > 0)
                mapped_to_any_feature |= is_mapped

                if feature_type == 'tiger_roads_feature':

                    not_on_road = feature_sets.map(lambda x: len(x) == 0)
                    crossed_paths_not_on_road |= not_on_road

                    not_on_primary_or_secondary_road = feature_sets.map(lambda x:
                        'primary_road' not in x and 'secondary_road' not in x)
                    crossed_paths_not_on_primary_or_secondary_road |= not_on_primary_or_secondary_road

            for unique_val in val_counts:
                if val_counts[unique_val] > int(1e6) or feature_type == 'safegraph_places_feature':
                    idxs = feature_sets.map(lambda x:unique_val in x).values
                    print("Adding feature %s, with %i rows, fraction %2.3f" % (unique_val, idxs.sum(), idxs.mean()))
                    key_to_store = 'feature_type=%s_%s' % (feature_type, unique_val)
                    key_to_store = key_to_store.replace(' ', '_')
                    stratification_idxs[key_to_store] = idxs
                    for combined_feature in feature_to_combined_features[key_to_store]:
                        combined_features[combined_feature] |= idxs
        new_features = [
            'feature_type=not_mapped_to_any_feature',
            'feature_type=crossed_paths_not_on_road',
            'feature_type=crossed_paths_not_on_primary_or_secondary_road']
        stratification_idxs['feature_type=not_mapped_to_any_feature'] = ~mapped_to_any_feature
        stratification_idxs['feature_type=crossed_paths_not_on_road'] = crossed_paths_not_on_road
        stratification_idxs['feature_type=crossed_paths_not_on_primary_or_secondary_road'] = crossed_paths_not_on_primary_or_secondary_road

        for home_census_feature in ['in_same_census_tract', 'outside_census_tract', 'visiting_census_tract']:
            is_mapped = all_path_crossings['home_census_feature'].map(lambda x: home_census_feature in json.loads(x))

            feature_name = 'feature_type=crossed_paths_not_on_road_and_%s' % home_census_feature
            stratification_idxs[feature_name] = crossed_paths_not_on_road & is_mapped
            new_features.append(feature_name)

            feature_name = 'feature_type=crossed_paths_not_on_primary_or_secondary_road_and_%s' % home_census_feature
            stratification_idxs[feature_name] = crossed_paths_not_on_primary_or_secondary_road & is_mapped
            new_features.append(feature_name)

        for combined_feature, idxs in combined_features.items():
            feature_name = 'feature_type=combined_feature_%s' % combined_feature
            stratification_idxs[feature_name] = idxs
            new_features.append(feature_name)

        for feature_type in new_features:
            print("Adding feature %s, with %i rows, fraction %2.3f" % (
                feature_type, stratification_idxs[feature_type].sum(), 
                stratification_idxs[feature_type].mean()))

    elif variable_to_stratify_by == 'local_hour':
        stratification_idxs = {}
        all_path_crossings['binned_local_hour'] = all_path_crossings['a_min_local_datetime'].map(lambda x:bin_local_hour(int(x.split()[1].split(':')[0])))
        for val in sorted(list(set(all_path_crossings['binned_local_hour']))):
            idxs = all_path_crossings['binned_local_hour'].values == val
            stratification_idxs['local_hour=%s' % val] = idxs
    else:
        raise Exception("invalid stratification variable %s" % variable_to_stratify_by)

    os.makedirs(STRATIFIED_PATH_CROSSINGS_DIR, exist_ok=True)
    for k in stratification_idxs:
        assert ' ' not in k
        outfile = os.path.join(STRATIFIED_PATH_CROSSINGS_DIR, 'path_crossings_%s.csv.gz' % k)
        safegraph_places_match = re.match(r'feature_type=safegraph_places_feature_(.+)', k)
        if safegraph_places_match:
            naics6_code = feature_name_to_naics6[safegraph_places_match.group(1)]
            df_to_write_out = all_path_crossings.loc[
                stratification_idxs[k],
                ['a_safegraph_id', 'b_safegraph_id', 'safegraph_places_feature']
            ].groupby(['a_safegraph_id', 'b_safegraph_id']).agg([
                'size',
                 lambda x: json.dumps(sorted(list(set(c for a in x for b in json.loads(a) if b[0] == naics6_code for c in b[1]))))
            ])
            df_to_write_out.columns = ['num_path_crossings_within_feature', 'individual_poi_ids']
        else:
            df_to_write_out = all_path_crossings.loc[
                stratification_idxs[k],
                ['a_safegraph_id', 'b_safegraph_id']
            ].groupby(['a_safegraph_id', 'b_safegraph_id']).size()
            df_to_write_out.name = 'num_path_crossings_within_feature'
        df_to_write_out = df_to_write_out.reset_index()
        df_to_write_out = df_to_write_out.join(temporal_path_crossings,
            on=['a_safegraph_id', 'b_safegraph_id'], how='left')
        df_to_write_out.to_csv(outfile, compression='gzip')
        print("Wrote out %s: %i path crossings, %i pairs" % (k, stratification_idxs[k].sum(), len(df_to_write_out)))

    print("Done writing out outfiles for %s" % variable_to_stratify_by)
    print("Peak memory usage: %d KB" % get_process_peak_memory_usage_kb())
    print("Total time: %d seconds" % (time.time() - t0))

def write_out_temporal_path_crossings():
    """
    Pre-computes temporal path crossings for all path crossings,
    this allows us to more rapidly compute segregation.
    """
    t_start = time.time()
    home_locations = dataprocessor.load_all_home_locations()
    path_crossings = load_path_crossings(use_dask=True, usecols=
        lambda x: x in ['a_safegraph_id', 'b_safegraph_id', 'min_utc_timestamp', 'a_min_local_datetime', 'median_latitude', 'median_longitude', 'min_dist']
        or x.startswith('satisfies_'))
    path_crossings = compute_temporal_path_crossings(path_crossings, home_locations)
    outfile = os.path.join(PATH_CROSSINGS_DIR, 'temporal_path_crossings.csv.gz')
    path_crossings.to_csv(outfile, compression='gzip')
    print('Computed temporal path crossings in %.2f seconds' % (time.time() - t_start))

def compute_temporal_path_crossings(path_crossings, home_locations, consolidation_interval=PATH_CROSSING_CONSOLIDATION_INTERVAL, stationary_thresholds=PATH_CROSSING_STATIONARY_THRESHOLDS):
    """
    Given a dataframe of path crossings, create a dataframe of unique user ID pairs with the following fields:
    - num_consecutive_in_group: For a given path crossing, how many path crossings are in the group
      where consecutive path crossings are grouped together.
    - num_groups: For a given path crossing, how many groups the safegraph id pair
      appears in. This value is the same for all path crossings for a given safegraph ID pair.
    - num_unique_days: For a given path crossing, how many unique local days the safegraph id pair
      appears in. This value is the same for all path crossings for a given safegraph ID pair.
    - same_home: Whether users live in the same home.
    """
    path_crossings = path_crossings.assign(
        local_day = path_crossings['a_min_local_datetime'].str.slice(stop=10),
        consolidation_key = path_crossings['min_utc_timestamp'].floordiv(consolidation_interval),
    )
    threshold_cols = ['a_safegraph_id', 'b_safegraph_id'] + [x for x in path_crossings.columns if x.startswith('satisfies_')]
    gb = path_crossings[threshold_cols].groupby(['a_safegraph_id', 'b_safegraph_id'])
    temporal_path_crossings = gb.max()
    path_crossings = path_crossings.drop_duplicates(['a_safegraph_id', 'b_safegraph_id', 'consolidation_key'])
    path_crossings = path_crossings.sort_values(['a_safegraph_id', 'b_safegraph_id', 'consolidation_key']).reset_index(drop=True)

    last_path_crossing = path_crossings.shift()

    dists_from_last_position = compute_distance_between_two_lat_lons(
        lat1=path_crossings['median_latitude'].values,
        lat2=last_path_crossing['median_latitude'].values,
        lon1=path_crossings['median_longitude'].values,
        lon2=last_path_crossing['median_longitude'].values,
    )
    dists_from_last_position[0] = 0

    for st in [None] + stationary_thresholds:
        col = 'num_consecutive_in_group'
        start_new_consecutive_group = (
            (last_path_crossing['a_safegraph_id'] != path_crossings['a_safegraph_id'])
            | (last_path_crossing['b_safegraph_id'] != path_crossings['b_safegraph_id'])
            | (last_path_crossing['consolidation_key']+1 != path_crossings['consolidation_key'])
        )
        if st is not None:
            col = f'num_consecutive_stationary_{st}m'
            start_new_consecutive_group |= (dists_from_last_position > st)
        path_crossings_th = path_crossings.assign(temporal_key=start_new_consecutive_group.cumsum())
        temporal_path_crossings[col] = path_crossings_th.groupby(['a_safegraph_id', 'b_safegraph_id', 'temporal_key']).size().groupby(level=[0,1]).max()

    groupby_pair = path_crossings.groupby(['a_safegraph_id', 'b_safegraph_id'])
    temporal_path_crossings['num_groups'] = groupby_pair.size()
    temporal_path_crossings['num_unique_days'] = groupby_pair['local_day'].nunique()
    temporal_path_crossings['min_dist'] = groupby_pair['min_dist'].min()

    num_interactors = path_crossings[['a_safegraph_id', 'b_safegraph_id', 'consolidation_key']]
    num_interactors_per_user = (
        num_interactors
        .append(
            num_interactors.rename(columns={'a_safegraph_id': 'b_safegraph_id', 'b_safegraph_id': 'a_safegraph_id'}),
            sort=False, ignore_index=True
        )
        .rename(columns={'a_safegraph_id': 'safegraph_id'})
        .groupby(['safegraph_id', 'consolidation_key'])
        .size()
    )
    num_interactors = num_interactors.join(
        num_interactors_per_user.rename('a_interactors_in_interval'),
        on=['a_safegraph_id', 'consolidation_key'],
        how='left',
    ).join(
        num_interactors_per_user.rename('b_interactors_in_interval'),
        on=['b_safegraph_id', 'consolidation_key'],
        how='left',
    ).assign(
        sum_interactors_in_interval=lambda x: x['a_interactors_in_interval']+x['b_interactors_in_interval']
    ).groupby(['a_safegraph_id', 'b_safegraph_id'])
    temporal_path_crossings['min_a_interactors_in_interval'] = num_interactors['a_interactors_in_interval'].min()
    temporal_path_crossings['min_b_interactors_in_interval'] = num_interactors['b_interactors_in_interval'].min()
    temporal_path_crossings['min_sum_interactors_in_interval'] = num_interactors['sum_interactors_in_interval'].min()

    temporal_path_crossings = temporal_path_crossings.reset_index()

    home_coordinates = home_locations[['safegraph_id', 'zillow_lat', 'zillow_lon']].set_index('safegraph_id')
    a_homes = temporal_path_crossings[['a_safegraph_id']].join(home_coordinates, on='a_safegraph_id')[['zillow_lat', 'zillow_lon']]
    b_homes = temporal_path_crossings[['b_safegraph_id']].join(home_coordinates, on='b_safegraph_id')[['zillow_lat', 'zillow_lon']]
    temporal_path_crossings['same_home'] = ((a_homes - b_homes).abs() < 1e-6).all(axis=1)

    return temporal_path_crossings

def analyze_one_measure_of_socioeconomic_segregation_using_path_crossings(ses_measure, all_home_locations, path_crossings, outfile):
    home_locations_under_analysis = all_home_locations.dropna(subset=[ses_measure])

    home_locations_under_analysis = home_locations_under_analysis.rename(
        columns={ses_measure: 'ses', 'new_census_CBSA Title': 'msa'})[['safegraph_id', 'ses', 'msa']]
    home_locations_under_analysis.set_index('safegraph_id', inplace=True)

    num_path_crossings = len(path_crossings)

    safegraph_id_pairs = path_crossings[['a_safegraph_id', 'b_safegraph_id']].drop_duplicates()
    safegraph_id_pairs = safegraph_id_pairs.append(
        safegraph_id_pairs.rename(columns={'a_safegraph_id': 'b_safegraph_id', 'b_safegraph_id': 'a_safegraph_id'}),
        sort=False, ignore_index=True)
    assert len(safegraph_id_pairs) == 2 * num_path_crossings
    assert safegraph_id_pairs.index.is_monotonic_increasing

    print("Computing segregation using path crossings + %s" % ses_measure)

    t0_regrouping = time.time()

    safegraph_id_pairs = safegraph_id_pairs.join(
        home_locations_under_analysis, on='a_safegraph_id', how='left'
    ).rename(columns={'ses':'a_ses', 'msa': 'a_msa'})

    safegraph_id_pairs = safegraph_id_pairs.join(
        home_locations_under_analysis['ses'], on='b_safegraph_id', how='left'
    ).rename(columns={'ses':'b_ses'})

    safegraph_id_pairs.dropna(subset=['a_ses', 'b_ses'], inplace=True)

    grouped_d = safegraph_id_pairs.groupby('a_msa')
    print("Time to regroup data into path crossings form: %2.3f seconds" % (time.time() - t0_regrouping))

    segregation_measures = {}

    mixed_model_results = {'segregation_index':[], 'n':[], 'group':[], 'full_mixed_model_results':[], 'n_obs':[], 'total_distinct_people': [], 'total_path_crossings': []}
    for msa, small_d in grouped_d:
        t0 = time.time()
        x = []
        y = []
        for a_safegraph_id, crossings in small_d.groupby('a_safegraph_id'):
            x.append(crossings.iloc[0]['a_ses'])
            y.append(list(crossings['b_ses']))
        total_distinct_people = len(x)
        total_path_crossings = len(small_d)
        mixed_model_estimate, n_people, n_obs, full_mixed_model_results = estimate_path_crossing_segregation_with_mixed_model(
            x=x,
            y=y,
            rescale_x=True,
            rescale_y=True,
            max_people_to_fit_model_on=10000,
        )
        mixed_model_results['segregation_index'].append(mixed_model_estimate)
        mixed_model_results['full_mixed_model_results'].append(full_mixed_model_results)
        mixed_model_results['n'].append(n_people)
        mixed_model_results['n_obs'].append(n_obs)
        mixed_model_results['group'].append(msa)
        mixed_model_results['total_distinct_people'].append(total_distinct_people)
        mixed_model_results['total_path_crossings'].append(total_path_crossings)
        print("mixed model estimate: %2.3f; MSA %s; n people %i; n obs %i; time to compute %2.3f seconds" % (
            mixed_model_estimate, msa, n_people, n_obs, time.time() - t0))
    mixed_model_results = pd.DataFrame(mixed_model_results)
    segregation_measures['path_crossings_with_mixed_model+%s' % ses_measure] = mixed_model_results

    grouped_unique_pairs = safegraph_id_pairs.groupby('a_safegraph_id')
    x = grouped_unique_pairs['a_ses'].first().to_numpy()
    y_means = grouped_unique_pairs['b_ses'].mean().to_numpy()
    y_medians = grouped_unique_pairs['b_ses'].median().to_numpy()
    msa_names = grouped_unique_pairs['a_msa'].first().to_numpy()

    segregation_measures['path_crossings_with_simple_means+%s' % ses_measure] = (
        compute_segregation_index_grouping_by_variable(
            x=x, y=y_means, groups=msa_names, include_overall=False))
    segregation_measures['path_crossings_with_simple_medians+%s' % ses_measure] = (
        compute_segregation_index_grouping_by_variable(
            x=x, y=y_medians, groups=msa_names, include_overall=False))

    pickle.dump(segregation_measures, outfile)
    print("Done computing path crossing segregation for %s" % ses_measure)

def get_path_crossing_stratification_levels(variable_to_stratify_by):
    """
    Returns path crossing stratification levels for a given variable_to_stratify_by.
    For local_hour, we stratify by hours 0-3, 3-6, etc
    For feature_type, we stratify by OSM features, homes, etc
    """
    assert variable_to_stratify_by in ['local_hour', 'feature_type']
    stratification_levels =  [a for a in os.listdir(STRATIFIED_PATH_CROSSINGS_DIR) if variable_to_stratify_by in a]
    assert all(a.startswith('path_crossings_') and a.endswith('.csv.gz') for a in stratification_levels)
    stratification_levels = sorted([a.replace('path_crossings_', '').replace('.csv.gz', '') for a in stratification_levels])
    return stratification_levels

def compute_segregation_index(x, y):
    """
    x and y are vectors
    x should be a measure of a person's SES
    y should be a measure of the SES of the people they interact with
    Returns the correlation between the two (equivalent to the Neighborhood Sorting Index, NSI). 
    """
    assert np.isnan(x).sum() == 0
    assert np.isnan(y).sum() == 0
    assert not dtype_pandas_series(x)
    assert not dtype_pandas_series(y)
    return pearsonr(x, y)[0]

def compute_segregation_index_grouping_by_variable(x, y, groups, include_overall):
    """
    Given a vector groups of the same length of x + y, computes the segregation for each group using compute_segregation_index.  
    x and y are vectors
    x should be a measure of a person's SES
    y should be a measure of the SES of the people they interact with
    groups is a vector of strings (eg, MSAs or other location groupings) of the same length as x + y. 
    """
    assert not dtype_pandas_series(groups)
    assert not dtype_pandas_series(x)
    assert not dtype_pandas_series(y)
    assert len(x) == len(y) == len(groups)
    segregation_indices = {'n': [], 'segregation_index': [], 'group': []}
    if include_overall:
        segregation_indices['n'].append(len(x))
        segregation_indices['segregation_index'].append(compute_segregation_index(x, y))
        segregation_indices['group'].append('overall')
    df = pd.DataFrame({'x':x, 'y':y, 'group':groups})
    for group, small_d in df.groupby('group'):
        if len(small_d) > 1:
            segregation_index_for_group = compute_segregation_index(small_d['x'].values, small_d['y'].values)
        else:
            segregation_index_for_group = np.nan
        segregation_indices['n'].append(len(small_d))
        segregation_indices['group'].append(group)
        segregation_indices['segregation_index'].append(segregation_index_for_group)
    return pd.DataFrame(segregation_indices).sort_values(by='segregation_index')[::-1]

def load_pings(utc_day_prefix='', max_to_load=None, use_dask=False, usecols=None, pick_random=False):
    """
    Load pings for analysis.
    """
    t_start = time.time()
    tuples = [(utc_day, utc_hour, id_prefix)
        for utc_day, utc_hour, id_prefix in itertools.product(UTC_DAYS_TO_USE_IN_ANALYSIS, VALID_UTC_HOURS, VALID_ID_PREFIXES)
        if utc_day.startswith(utc_day_prefix)]
    original_file_count = len(tuples)
    if max_to_load is not None:
        if pick_random:
            random.seed(4987133829387)
            tuples = random.sample(tuples, max_to_load)
        else:
            tuples = tuples[:max_to_load]
    print("Loading %i out of %i files" % (len(tuples), original_file_count))
    filenames = [
        os.path.join(*generate_filepath('locations', id_prefix, utc_day, utc_hour))
        for utc_day, utc_hour, id_prefix in tuples
    ]
    d = load_ping_file(file_path=filenames, use_dask=use_dask, usecols=usecols)
    print("Loaded %d pings in %2.3fs" % (len(d), time.time() - t_start))
    return d

def load_path_crossings(utc_day_prefix='', utc_hours_to_use=None, max_to_load=None, use_dask=False, usecols=None, annotated=False, pick_random=False, get_outfile_fn=None):
    """
    Read in path crossings files.
    """
    t_start = time.time()
    if utc_hours_to_use is None:
        utc_hours_to_use = VALID_UTC_HOURS
    utc_days_hours = [(utc_day, utc_hour)
        for utc_day, utc_hour in itertools.product(UTC_DAYS_TO_USE_IN_ANALYSIS, VALID_UTC_HOURS)
        if (utc_day.startswith(utc_day_prefix) and (utc_hour in utc_hours_to_use))]

    if max_to_load is not None:
        if pick_random:
            random.seed(239801242309)
            utc_days_hours = random.sample(utc_days_hours, max_to_load)
        else:
            utc_days_hours = utc_days_hours[:max_to_load]
    print("%i files to load for path crossings" % len(utc_days_hours))

    if get_outfile_fn is None:
        get_outfile_fn = dataprocessor.get_UNFILTERED_path_crossings_outfile

    path_crossings_filenames = [get_outfile_fn(utc_day, utc_hour, annotated=annotated) for utc_day, utc_hour in utc_days_hours]

    path_crossings = load_csv_possibly_with_dask(path_crossings_filenames, use_dask=use_dask,
        usecols=usecols).drop_duplicates()

    print("After removing identical time/user pairs, %i rows" % len(path_crossings))
    print("Seconds for loading path crossings: %2.3f" % (time.time() - t_start))
    print("Peak memory usage: %d KB" % get_process_peak_memory_usage_kb())
    return path_crossings

def map_path_crossings_to_features():
    """
    Load path crossings files and annotate using CombinedPOIMapper. 
    """
    poi_sources_to_use = ALL_FEATURE_MAPPING_DATA_SOURCES
    combined_poi_mapper = dataprocessor.CombinedPOIMapper(source_names=poi_sources_to_use, use_small_prototyping_datasets=False)

    utc_days_hours = list(itertools.product(UTC_DAYS_TO_USE_IN_ANALYSIS, VALID_UTC_HOURS))
    for i, (utc_day, utc_hour) in enumerate(utc_days_hours):
        outfile_name = dataprocessor.get_UNFILTERED_path_crossings_outfile(utc_day, utc_hour, annotated=True)
        os.makedirs(os.path.dirname(outfile_name), exist_ok=True)
        outfile = open_file_for_exclusive_writing(outfile_name)
        if outfile is None: continue
        with outfile:
            previous_hour_path_crossings = None
            if i > 0:
                previous_hour_path_crossings = load_path_crossings(utc_day_prefix=utc_days_hours[i-1][0], utc_hours_to_use=[utc_days_hours[i-1][1]])

            print("Mapping path crossings for %s hour %s to features" % (utc_day, utc_hour))
            t0 = time.time()
            path_crossings = load_path_crossings(utc_day_prefix=utc_day, utc_hours_to_use=[utc_hour])

            if previous_hour_path_crossings is not None:
                path_crossing_identifier_columns = ['a_safegraph_id', 'b_safegraph_id', 'min_utc_timestamp']
                rows_to_drop = path_crossings[path_crossing_identifier_columns].merge(
                    previous_hour_path_crossings[path_crossing_identifier_columns],
                    on=path_crossing_identifier_columns,
                    how='left',
                    indicator=True)._merge == 'both'
                print('For %s hour %s, dropping %d/%d path crossings from previous hour' %
                    (utc_day, utc_hour, rows_to_drop.sum(), len(path_crossings)))
                path_crossings = path_crossings[~rows_to_drop]

            previous_hour_path_crossings = path_crossings

            result = combined_poi_mapper.query(
                lat=path_crossings['median_latitude'].values,
                lon=path_crossings['median_longitude'].values)

            assert len(result) == len(path_crossings)
            assert np.allclose(result['latitude'].values, path_crossings['median_latitude'].values)
            assert np.allclose(result['longitude'].values, path_crossings['median_longitude'].values)
            for k in poi_sources_to_use + ['any']:
                path_crossings['%s_feature' % k] = result['%s_feature' % k].values
            temp_outfile_name = outfile_name + '.tmp'
            path_crossings.to_csv(temp_outfile_name, compression='gzip')
            shutil.copyfileobj(open(temp_outfile_name, 'rb'), outfile)
            os.unlink(temp_outfile_name)
            print("Successfully wrote out annotated path crossings outfile outfile to %s; total time taken, %2.3f seconds" % (outfile_name, time.time() - t0))

def run_all_jobs_for_analyze_one_measure_of_socioeconomic_segregation_using_path_crossings(variable_to_stratify_by, compute_temporal):
    all_home_locations = dataprocessor.load_all_home_locations()
    ses_measures = SES_MEASURES_FOR_ECONOMIC_SEGREGATION
    stratification_levels = [None]
    if variable_to_stratify_by is not None:
        stratification_levels = get_path_crossing_stratification_levels(variable_to_stratify_by)

    all_jobs = []

    JobSpec = namedtuple('JobSpec', 'ses_measure stratification_key field field_min no_same_home filename')

    for ses_measure, stratification_key in itertools.product(ses_measures, stratification_levels):
        strat_filename_part = ''
        if stratification_key is not None:
            strat_filename_part = '_' + stratification_key
        all_jobs.append(JobSpec(ses_measure, stratification_key, field=None, field_min=None,
            no_same_home=False,
            filename='all_path_crossing_ses_measure_%s%s.pkl' % (ses_measure, strat_filename_part)))
        all_jobs.append(JobSpec(ses_measure, stratification_key, field=None, field_min=None,
            no_same_home=True,
            filename='all_path_crossing_ses_measure_%s%s_no_same_home.pkl' % (ses_measure, strat_filename_part)))

        if compute_temporal:
            for field, field_min in TEMPORAL_PATH_CROSSING_THRESHOLDS:
                all_jobs.append(JobSpec(ses_measure, stratification_key, field, field_min,
                    no_same_home=False,
                    filename='all_path_crossing_ses_measure_%s%s_min_%s_%d.pkl' % (ses_measure, strat_filename_part, field, field_min)))
                all_jobs.append(JobSpec(ses_measure, stratification_key, field, field_min,
                    no_same_home=True,
                    filename='all_path_crossing_ses_measure_%s%s_no_same_home_min_%s_%d.pkl' % (ses_measure, strat_filename_part, field, field_min)))

    last_stratification_key = None
    path_crossings = None
    for current_idx, (ses_measure, stratification_key, field, field_min, no_same_home, filename) in enumerate(all_jobs):
        print("Running job %d/%d SES measure %s stratification key %s no same home %s with min %s = %s" % (
            current_idx+1, len(all_jobs), ses_measure, stratification_key, no_same_home, field, field_min))
        outfile_name = os.path.join(ECONOMIC_SEGREGATION_MEASURES_DIR, filename)
        os.makedirs(os.path.dirname(outfile_name), exist_ok=True)
        outfile = open_file_for_exclusive_writing(outfile_name)
        if outfile is None: continue
        with outfile:
            if path_crossings is None or stratification_key != last_stratification_key:
                print('Loading path crossings')
                if stratification_key is None:
                    path_crossings = pd.read_csv(os.path.join(PATH_CROSSINGS_DIR, 'temporal_path_crossings.csv.gz'), index_col=0)
                else:
                    path_crossings = pd.read_csv(os.path.join(STRATIFIED_PATH_CROSSINGS_DIR, 'path_crossings_%s.csv.gz' % stratification_key))
                last_stratification_key = stratification_key
            path_crossings_to_use = path_crossings
            if field is not None:
                path_crossings_to_use = path_crossings_to_use[path_crossings_to_use[field] >= field_min]
            if no_same_home:
                path_crossings_to_use = path_crossings_to_use[path_crossings_to_use['same_home'] == False]
            print("Computing segregation with %i path crossings" % len(path_crossings_to_use))
            analyze_one_measure_of_socioeconomic_segregation_using_path_crossings(
                ses_measure=ses_measure,
                path_crossings=path_crossings_to_use,
                all_home_locations=all_home_locations,
                outfile=outfile)
            path_crossings_to_use = None

if __name__ == '__main__':
    pd.set_option('max_columns', 50)
    pd.set_option('max_rows', 500)

    functions = [
        'analyze_one_measure_of_socioeconomic_segregation_using_path_crossings_no_stratification', 
        'analyze_one_measure_of_socioeconomic_segregation_using_path_crossings_with_stratification',
        'map_path_crossings_to_features',
        'write_out_stratified_path_crossings',
        'write_out_temporal_path_crossings',
        'write_out_path_crossings_for_visualization'
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('function_to_run', choices=functions)

    initial_args = sys.argv[1:2]
    secondary_args = sys.argv[2:]
    init_args = parser.parse_args(initial_args)

    parser = argparse.ArgumentParser()
    if init_args.function_to_run == 'analyze_one_measure_of_socioeconomic_segregation_using_path_crossings_no_stratification':
        parser.add_argument('--compute-temporal', action='store_true')

        args = parser.parse_args(secondary_args)
        run_all_jobs_for_analyze_one_measure_of_socioeconomic_segregation_using_path_crossings(
            variable_to_stratify_by=None,
            compute_temporal=args.compute_temporal)
    elif init_args.function_to_run == 'analyze_one_measure_of_socioeconomic_segregation_using_path_crossings_with_stratification':
        parser.add_argument('variable_to_stratify_by', choices=['local_hour', 'feature_type'])
        parser.add_argument('--compute-temporal', action='store_true')
        args = parser.parse_args(secondary_args)
        run_all_jobs_for_analyze_one_measure_of_socioeconomic_segregation_using_path_crossings(
            variable_to_stratify_by=args.variable_to_stratify_by,
            compute_temporal=args.compute_temporal)
    elif init_args.function_to_run == 'map_path_crossings_to_features':
        map_path_crossings_to_features()
    elif init_args.function_to_run == 'write_out_stratified_path_crossings':
        parser.add_argument('variable_to_stratify_by')
        args = parser.parse_args(secondary_args)
        write_out_stratified_path_crossings(args.variable_to_stratify_by)
    elif init_args.function_to_run == 'write_out_temporal_path_crossings':
        write_out_temporal_path_crossings()
