# exposure-segregation

Code to generate results in [Human mobility networks reveal increased segregation in large cities](https://arxiv.org/abs/2210.07414)

## System requirements

Most experiments in the paper were performed using a server running Ubuntu 16.04 LTS x86_64 with 288 threads and 12 TB RAM. For the simulated data, a standard computer with 64 GB RAM (or possibly 32 GB) should be sufficient.

The overall amount of time required for the pipeline is roughly several hours for the simulated data and several days for the real data. Many steps can be parallelized by running the script multiple times in parallel, which will significantly speed up the computation.

## Computing basic results

1. **Setting up virtualenv.** Our code is run in a conda environment on Linux. Install python 3.7 [miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers) and set up this environment by running `conda env create --prefix YOUR_PATH_HERE --file safegraph_env_v3.yml`. Once you have set up the environment, activate it prior to running any code by running `source YOUR_PATH_HERE/bin/activate`. 

2. **Run unit tests.** We have provided unit tests for some critical code and statistical models. The tests will ensure that your environment is set up correctly, especially the R/Python rpy2 interface which is required for our mixed model.

* `python analysis_unit_tests.py`

3. **Generate simulated data.** The analysis depends on non-public data, but we have provided a script to generate simulated data to run the code. Of course, real SafeGraph data can be used instead of the simulated data.

* `python generate_simulated_data.py`

4. **Extract middle-of-night locations for users and infer home locations.** This extracts middle-of-night locations for users and computes home locations for users who have sufficiently consistent middle-of-night locations. This can be run several times in parallel and it will figure out how to distribute the work automatically.

* `python dataprocessor.py extract_middle_of_night_locations`

5. **Link inferred home locations to more precise inferred income information from both Zillow and the 5-Year ACS.**

* Download [ACS data](https://www2.census.gov/geo/tiger/TIGER_DP/2017ACS/ACS_2017_5YR_BG.gdb.zip) and unzip to base_dir.
* Download [county to MSA mapping](https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2017/delineation-files/list1.xls), convert to CSV, and save as august_2017_county_to_metropolitan_mapping.csv in base_dir.
* Download [cbsa-est2018-alldata.csv](https://www2.census.gov/programs-surveys/popest/datasets/2010-2018/metro/totals/cbsa-est2018-alldata.csv) to base_dir.
* `python dataprocessor.py create_final_zestimate_files`

6. **Compute duplicate pings.** As described in the paper, we remove users where > 80% of their pings are duplicates with another user.

* `python dataprocessor.py compute_duplicate_pings` (This can be run several times in parallel and it will figure out how to distribute the work automatically.)
* `python dataprocessor.py compute_fraction_of_duplicate_pings_per_user` (Cannot be parallelized.)

7. **Compute pairs of users whose paths cross.** This can be run several times in parallel and it will figure out how to distribute the work automatically.

* `python dataprocessor.py compute_users_whose_paths_cross`

8. **Write out temporal path crossings.** This consolidates the path crossings into one row per distinct pair of people, which is used to compute exposure segregation.

* `python analysis.py write_out_temporal_path_crossings`

9. **Compute exposure segregation.** The mixed model is run on different scenarios. For example, all path crossings, all path crossings excluding people living in the same home, path crossings for people who met on at least N distinct days, etc. More details can be seen in the code. This can be run several times in parallel and it will figure out how to distribute the work automatically.

* `python analysis.py analyze_one_measure_of_socioeconomic_segregation_using_path_crossings_no_stratification`
* The `--compute-temporal` flag can be added to compute additional scenarios. See the code for more details.

10. **Explore exposure segregation results.** Here is some example code on how to load the results. Read the code for more information about the parameters. Each file corresponds to different scenarios, as described above.

* `msa_census_info = dataprocessor.get_msa_census_info()`
* `dataprocessor.load_path_crossing_segregation_file('all_path_crossing_ses_measure_rent_zestimate.pkl', 'metro', msa_census_info)`

Here is the expected output. With simulated data, the exposure segregation is all 1.0. Using real data, the output would reflect the results in the paper.

```
segregation_index                          group    full_mixed_model_results ...
              1.0                   Beatrice, NE      {...}
              1.0                Blytheville, AR
              1.0                    Durango, CO
              1.0                       Elko, NV
              1.0   Lake Havasu City-Kingman, AZ
              1.0                  Las Vegas, NM
              1.0                  Maryville, MO
              1.0    Omaha-Council Bluffs, NE-IA
              1.0                          Other
              1.0                   Riverton, WY
```

## Computing results stratified by local hour / feature (POI) type

These results additionally require a non-public SafeGraph Places dataset for the POI mapping. You can remove them from `ALL_FEATURE_MAPPING_DATA_SOURCES` in constants_and_util.py to generate the results that do not require them.

1. **Map path crossings to features.** This can be run several times in parallel and it will figure out how to distribute the work automatically.

* Download [tiger rails dataset](https://www2.census.gov/geo/tiger/TGRGDB17/tlgdb_2017_a_us_rails.gdb.zip) and extract to base_dir.
* Download [tiger roads dataset](https://www2.census.gov/geo/tiger/TGRGDB17/tlgdb_2017_a_us_roads.gdb.zip) and extract to base_dir.
* `python analysis.py map_path_crossings_to_features`

2. **Write out stratified path crossings.** This produces one path crossings file for each local hour / feature (POI) type.

* `python analysis.py write_out_stratified_path_crossings local_hour`
* `python analysis.py write_out_stratified_path_crossings feature_type`

3. **Compute exposure segregation.** This runs basically the same code as `analyze_one_measure_of_socioeconomic_segregation_using_path_crossings_no_stratification` above, but stratifying by local hour / feature (POI) type.

* `python analysis.py analyze_one_measure_of_socioeconomic_segregation_using_path_crossings_with_stratification local_hour`
* `python analysis.py analyze_one_measure_of_socioeconomic_segregation_using_path_crossings_with_stratification feature_type`
* The `--compute-temporal` flag can be added to compute additional scenarios. See the code for more details.
