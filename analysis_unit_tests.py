import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import analysis
import io
import random

def basic_unit_tests_for_compute_temporal_path_crossings():
    path_crossings = pd.DataFrame([
        ['a', 'b', 0, '2000-01-01 00:00:00', 37.558, -122.149, 5.6],
        ['a', 'b', 8, '2000-01-01 00:00:00', 37.559, -122.148, 3.1],
        ['a', 'b', 9, '2000-01-01 00:00:00', 37.559, -122.148, 4.3],
        ['a', 'b', 14, '2000-01-01 00:00:00', 37.569, -122.155, 1.2],
        ['a', 'c', 0, '2000-01-01 00:00:00', 37.558, -122.149, 1.3],
        ['a', 'c', 100, '2000-01-01 00:00:00', 37.558, -122.149, 6.6],
        ['a', 'd', 2, '2000-01-01 00:00:00', 37.558, -122.149, 1.4],
        ['b', 'c', 55, '2000-01-01 00:00:00', 37.558, -122.149, 9.4],
        ['b', 'c', 100000, '2000-01-02 00:00:00', 37.558, -122.149, 7.6],
        ['b', 'c', 200000, '2000-01-03 00:00:00', 37.558, -122.149, 8.7],
    ], columns=[
        'a_safegraph_id',
        'b_safegraph_id',
        'min_utc_timestamp',
        'a_min_local_datetime',
        'median_latitude',
        'median_longitude',
        'min_dist',
    ])
    home_locations = pd.DataFrame({
        'safegraph_id': ['a', 'b', 'c', 'd'],
        'zillow_lat': [1, 1, 1, 10],
        'zillow_lon': [4, 5, 4, 10],
    })
    temporal_path_crossings = analysis.compute_temporal_path_crossings(path_crossings, home_locations, 5, [500, 5000])
    expected_temporal_path_crossings = pd.DataFrame([
        ['a', 'b', 3, 2, 3, 3, 1, 1.2, 1, 1, 2, False],
        ['a', 'c', 1, 1, 1, 2, 1, 1.3, 1, 1, 2, True],
        ['a', 'd', 1, 1, 1, 1, 1, 1.4, 3, 1, 4, False],
        ['b', 'c', 1, 1, 1, 3, 3, 7.6, 1, 1, 2, False],
    ], columns=[
        'a_safegraph_id',
        'b_safegraph_id',
        'num_consecutive_in_group',
        'num_consecutive_stationary_500m',
        'num_consecutive_stationary_5000m',
        'num_groups',
        'num_unique_days',
        'min_dist',
        'min_a_interactors_in_interval',
        'min_b_interactors_in_interval',
        'min_sum_interactors_in_interval',
        'same_home'
    ])
    pd.testing.assert_frame_equal(temporal_path_crossings, expected_temporal_path_crossings)
    print('Passed basic unit tests for computing temporal path crossings')

def basic_unit_tests_on_mixed_model():
    """
    Make sure a bunch of things we expect to be true are actually true in simulated data. 
    Also checked that we correctly recover alpha and noise_1 in simulated data, although this is not included in final code. 
    Also checked that, as n_people -> infinity, it seems like we recover the exact correct segregation estimates (error drops below 0.01 in all conditions with n_people=20000)
    """
    print("basic unit tests on mixed model")

    path_crossings = pd.DataFrame({
        'a_safegraph_id': ['a', 'a', 'a', 'b', 'c'],
        'b_safegraph_id': ['b', 'c', 'd', 'd', 'd'],
        'other_data': [5, 4, 3, 2, 1],
    })
    home_locations = pd.DataFrame({
        'safegraph_id': ['a', 'b', 'c'],
        'dummy_ses_measure': [1, 2, 3],
        'new_census_Metropolitan/Micropolitan Statistical Area': 'Metropolitan Statistical Area',
        'new_census_CBSA Title': ['msa1', 'msa1', 'msa2'],
    })

    path_crossings_original = path_crossings.copy()
    home_locations_original = home_locations.copy()

    analysis.analyze_one_measure_of_socioeconomic_segregation_using_path_crossings('dummy_ses_measure', home_locations, path_crossings, io.BytesIO())
    pd.testing.assert_frame_equal(path_crossings, path_crossings_original)
    pd.testing.assert_frame_equal(home_locations, home_locations_original)

    np.random.seed(0)
    random.seed(0)

    a = 1
    max_points_per_person = 10
    noise_scale_2 = 5
    b = -.5
    noise_scale_1 = 10
    n_people = 1000
    x_scale = 2
    all_kwargs = {'a':a, 'b':b, 'noise_scale_1':noise_scale_1, 'noise_scale_2':noise_scale_2, 'n_people':n_people, 'max_points_per_person':max_points_per_person, 'x_scale':x_scale}
    x, y, true_segregation = simulate_data_to_test_mixed_model(**all_kwargs)

    atol = 0.001
    rtol = 0

    original_segregation_estimate = analysis.estimate_path_crossing_segregation_with_mixed_model(x, y, max_people_to_fit_model_on=1000, rescale_x=True, rescale_y=True)[0]
    print("True segregation is %2.6f" % true_segregation)
    print("Mixed model segregation is %2.6f" % original_segregation_estimate)
    np.testing.assert_allclose(true_segregation, 0.169967, atol=atol, rtol=rtol)
    np.testing.assert_allclose(actual=original_segregation_estimate, atol=atol, rtol=rtol,
        desired=0.177851169967)

    original_naive_estimate = estimate_path_crossing_segregation_with_naive_model(x, y)
    print("Naive segregation estimats: %2.6f" % original_naive_estimate)

    print("\n\n****Testing that it doesn't make a difference whether we rescale Y") # We just rescale y for computational reasons, not for correctness, so it shouldn't affect the estimate. 
    estimate2 = analysis.estimate_path_crossing_segregation_with_mixed_model(x, y, max_people_to_fit_model_on=1000, rescale_x=True, rescale_y=False)[0]
    np.testing.assert_allclose(original_segregation_estimate, estimate2, atol=atol, rtol=rtol)

    print("\n\n****Testing that it DOES make a difference whether we rescale X") # we would expect this to make a difference because mixed model code assumes x is rescaled to have zero mean and unit variance. 
    estimate3 = analysis.estimate_path_crossing_segregation_with_mixed_model(x, y, max_people_to_fit_model_on=1000, rescale_x=False, rescale_y=True)[0]
    assert not np.allclose(original_segregation_estimate, estimate3, atol=atol, rtol=rtol)

    print("\n\n****Testing that, when rescaling both X and Y, their scale makes no difference")
    estimate4 = analysis.estimate_path_crossing_segregation_with_mixed_model(
        x=list(np.array(x) * 50 - 2934), 
        y=[list(np.array(y_i) * 34 + 498) for y_i in y], 
        max_people_to_fit_model_on=1000,
        rescale_x=True, 
        rescale_y=True)[0]
    np.testing.assert_allclose(original_segregation_estimate, estimate4, atol=atol, rtol=rtol)

    all_results = []
    for a in [1]:
        for max_points_per_person in [20, 10, 5]:
            for noise_scale_1 in [2, 5, 10]:
                for noise_scale_2 in [5, 10]:
                    b = -.5
                    n_people = 1000
                    x_scale = 2
                    all_kwargs = {'a':a, 'b':b, 'noise_scale_1':noise_scale_1, 'noise_scale_2':noise_scale_2, 'n_people':n_people, 'max_points_per_person':max_points_per_person, 'x_scale':x_scale}
                    print("\n\nSimulated data kwargs")
                    print(all_kwargs)
                    x, y, true_segregation = simulate_data_to_test_mixed_model(**all_kwargs)

                    mixed_model_estimate, n_people, n_obs, mixed_model_results = analysis.estimate_path_crossing_segregation_with_mixed_model(x, y, max_people_to_fit_model_on=1000, rescale_x=True, rescale_y=True)
                    naive_estimate = estimate_path_crossing_segregation_with_naive_model(x, y)
                    print("True segregation: %2.5f; mixed model estimate %2.5f; naive estimates %2.5f" % (true_segregation, mixed_model_estimate, naive_estimate))
                    all_results.append({'true_segregation':true_segregation, 'mixed_model_estimate':mixed_model_estimate, 'naive_estimate':naive_estimate})
    expected_results = [
        {   'mixed_model_estimate': 0.6989794770314164,
            'naive_estimate': 0.5715794669656461,
            'true_segregation': 0.7009388533949436},
        {   'mixed_model_estimate': 0.7465106497935253,
            'naive_estimate': 0.425418730961027,
            'true_segregation': 0.7310662522699028},
        {   'mixed_model_estimate': 0.3391817943314789,
            'naive_estimate': 0.3138507704588768,
            'true_segregation': 0.33710537966018495},
        {   'mixed_model_estimate': 0.3902390471034377,
            'naive_estimate': 0.3059001303810065,
            'true_segregation': 0.3761849888632058},
        {   'mixed_model_estimate': 0.19917516683427733,
            'naive_estimate': 0.19371292906342075,
            'true_segregation': 0.21335554922669642},
        {   'mixed_model_estimate': 0.20011666804803185,
            'naive_estimate': 0.1952860218011061,
            'true_segregation': 0.20111863225951573},
        {   'mixed_model_estimate': 0.7671716762322555,
            'naive_estimate': 0.5708799241656138,
            'true_segregation': 0.7142582758974351},
        {   'mixed_model_estimate': 0.654876120555361,
            'naive_estimate': 0.3268568949218779,
            'true_segregation': 0.7128403453943545},
        {   'mixed_model_estimate': 0.3841638210484421,
            'naive_estimate': 0.33783508060970685,
            'true_segregation': 0.38478234767396635},
        {   'mixed_model_estimate': 0.39096408955077017,
            'naive_estimate': 0.2855862599662394,
            'true_segregation': 0.3967871600458214},
        {   'mixed_model_estimate': 0.18796638144213917,
            'naive_estimate': 0.18393624010537188,
            'true_segregation': 0.17482820583954625},
        {   'mixed_model_estimate': 0.12988284226010335,
            'naive_estimate': 0.10853055991816646,
            'true_segregation': 0.15523239935642114},
        {   'mixed_model_estimate': 0.6695995584594858,
            'naive_estimate': 0.4581530323773302,
            'true_segregation': 0.6919054245853591},
        {   'mixed_model_estimate': 0.6767734910522172,
            'naive_estimate': 0.28128877868234925,
            'true_segregation': 0.7146229480196622},
        {   'mixed_model_estimate': 0.34399981421756937,
            'naive_estimate': 0.295138481816,
            'true_segregation': 0.32407071871374865},
        {   'mixed_model_estimate': 0.49021263180667196,
            'naive_estimate': 0.28553424633410324,
            'true_segregation': 0.43513800249244877},
        {   'mixed_model_estimate': 0.20808005509095373,
            'naive_estimate': 0.19776694182214682,
            'true_segregation': 0.21054898697245575},
        {   'mixed_model_estimate': 0.19812436562881544,
            'naive_estimate': 0.17511952780706477,
            'true_segregation': 0.2055889744255676},
    ]
    for i in range(len(all_results)):
        for k in expected_results[i].keys():
            np.testing.assert_allclose(expected_results[i][k], all_results[i][k], atol=atol, rtol=rtol)

    all_results = pd.DataFrame(all_results)
    print("Minimum true segregation: %2.3f" % all_results['true_segregation'].min())
    print("Maximum true segregation: %2.3f" % all_results['true_segregation'].max())
    print("Maximum difference between true and mixed model estimate: %2.3f" % np.abs(all_results['true_segregation'] - all_results['mixed_model_estimate']).max())
    max_val = np.max(all_results.values) * 1.05
    plt.figure(figsize=[16, 4])

    to_plot = ['naive_estimate', 'mixed_model_estimate']
    for i, estimate_name in enumerate(to_plot):
        plt.subplot(1, len(to_plot), i+1)
        plt.scatter(all_results['true_segregation'], all_results[estimate_name])
        plt.title(estimate_name)
        plt.xlabel("True segregation")
        plt.ylabel(estimate_name)
        plt.xlim([0, max_val])
        plt.ylim([0, max_val])
        plt.plot([0, max_val], [0, max_val], color='black')

    plt.savefig('mixed_model_tests.png', dpi=300)

def estimate_path_crossing_segregation_with_naive_model(x, y):
    """
    Estimates path crossing segregation using simple correlation.
    Correlation is computed between each user and the average SES of people they cross paths with.
    """
    mean_ses = [np.mean(vals) for vals in y]
    return analysis.compute_segregation_index(x, mean_ses)

def simulate_data_to_test_mixed_model(a, b, noise_scale_1, noise_scale_2, n_people, max_points_per_person, x_scale):
    """
    Simulate data according to mixed model assumptions to ensure we can (approximately) recover the correct parameters. 
    """
    x = np.random.randn(n_people) * x_scale
    individual_noise_term = np.random.randn(n_people) # this is the true value of e_1. 
    y = []
    mean_interactor_ses = a * x + b + individual_noise_term * noise_scale_1 # this is the true value of mu (unobserved). 
    true_segregation = pearsonr(mean_interactor_ses, x)[0]
    assert true_segregation > .1 # things get weird otherwise, so don't choose crazy parameter values. 
    for i in range(n_people):
        n_points = random.choice(range(1, max_points_per_person + 1))
        y.append(list(mean_interactor_ses[i] + np.random.randn(n_points) * noise_scale_2))
    return x, y, true_segregation

if __name__ == '__main__':
    basic_unit_tests_for_compute_temporal_path_crossings()
    basic_unit_tests_on_mixed_model()
