# covid-mobility-tool

Code to generate results in "Supporting COVID-19 policy response with large-scale mobility-based modeling" (2021) by Serina Chang, Mandy L. Wilson, Bryan Lewis, Zakaria Mehrab, Komal K. Dudakiya, Emma Pierson, Pang Wei Koh, Jaline Gerardin, Beth Redbird, David Grusky, Madhav Marathe, and Jure Leskovec. 

This is an updated version of our [covid-mobility](https://github.com/snap-stanford/covid-mobility) repo, which contains code corresponding to our first publication, "[Mobility network models of COVID-19 explain inequities and inform reopening](https://www.nature.com/articles/s41586-020-2923-3)" (Chang, Pierson, Koh, et al., *Nature* 2020).

## Regenerating results

1. **Setting up virtualenv**. Our code is run in a conda environment, with all analysis performed on a Linux Ubuntu system. You can set up this environment by running `conda env create --prefix YOUR_PATH_HERE --file safegraph_env_v3.yml`. Once you have set up the environment, activate it prior to running any code by running `source YOUR_PATH_HERE/bin/activate`. 

2. **Downloading datasets**. See Section 2.1 of our paper for details on the following datasets.
    - SafeGraph data is freely available to researchers, non-profits, and governments through the [SafeGraph COVID-19 Data Consortium](https://www.safegraph.com/covid-19-data-consortium). We use SafeGraph's [Places](https://docs.safegraph.com/v4.0/docs/places-schema) and [Weekly Patterns (v2)](https://docs.safegraph.com/v4.0/docs/weekly-patterns) datasets for fine-grained information about points of interest (POI), and SafeGraph's [Social Distancing Metrics](https://docs.safegraph.com/v4.0/docs/social-distancing-metrics) for mobility information per census block group (CBG).
    
    - We use mask-wearing data from the Institute for Health Metrics and Evaluation (IHME) website, available [here](https://covid19.healthdata.org/united-states-of-america/virginia?view=mask-use). We also use IHME's estimates from their January 2021 [briefing](http://www.healthdata.org/sites/default/files/files/Projects/COVID/2021/briefing_US_20210114.pdf) of the varying COVID-19 case detection rate over time in the US. 
    
    - We use COVID-19 reported cases and deaths from *The New York Times*, available [here](https://github.com/nytimes/covid-19-data).
    
    - Census data comes from the American Community Survey. CBG shapefiles, with linked data from the 5-year 2013-2017 ACS, are available [here](https://www2.census.gov/geo/tiger/TIGER_DP/2017ACS/ACS_2017_5YR_BG.gdb.zip). (We note that, as described in Section 2.1, we use the 1-year 2018 estimates for the current population of each CBG.) The mapping from counties to MSAs is available [here](https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2017/delineation-files/list1.xls). 

3. **Preparing data for model**. Before we can run models, we need to perform a series of processing steps on the data. The first two steps listed below must be run sequentially; the third step can be run at any point.
    - **Processing raw SafeGraph data.** We process SafeGraph Patterns data and produce data files per metropolitan statistical area (MSA) using `process_safegraph_data.ipynb` (some of the cell outputs are cleared here as to not reveal raw SafeGraph data). Our code makes references to specific filepaths in our system, but should be adaptable to other file structures by modifying macro paths in `covid_constants_and_util.py`.
    - **Generating hourly POI-CBG networks.** We apply the iterative proportional fitting procedure (IPFP) to infer hourly POI-CBG visit matrices from SafeGraph data (see Methods in our *Nature* paper for details). To infer the network for a given MSA, run `python generate_ipf.py MSA_NAME`. 
    - **Estimating initial SEIR states for model.** To begin a simulation on a date that is past the beginning of the pandemic (e.g., November 1, 2020), we use the historical reported COVID-19 cases and the time-varying case detection rate to estimate how many people should be initialized to each SEIR state in each CBG on that date (see Section A.1.3 in this paper for details). This estimation occurs in `initialize_seir.ipynb`. 

4. **Running models.** Models are run using `model_experiments.py`. Running all the models described in the paper is computationally expensive. Specifically, most experiments in the paper were performed using a server with 288 threads and 12 TB RAM; saving the models required several terabytes of disk space. (Tip: to run a step for a subset of MSAs, you can modify `MSAS_IMPLEMENTED_FOR_V2` in `covid_constants_and_util.py`.) The following steps should be run in order.
    - **Determining plausible ranges for model parameters.** We determine prior ranges for the free parameters by ensuring that simulations with those parameters produce reproductive numbers that match plausible values of R0 (see Methods in our *Nature* paper for details). Run `python model_experiments.py run_many_models_in_parallel calibrate_r0`. This will start 30 jobs per MSA.
    - **Estimating best-fit parameters.** We conduct grid search to find the parameter sets which best fit reported case counts. Run `python model_experiments.py run_many_models_in_parallel normal_grid_search`. This will start 1050 jobs per MSA.  
    - **Use case experiments.** The use case experiments rely on having grid search completed, since they use the best-fit model parameters. All experiments can be run using the same call signature: `python model_experiments.py run_many_models_in_parallel EXPERIMENT_NAME --how_to_select_best_grid_search_models smoothed_daily_cases_rmse_time_varying_cdr`. In this paper, we focus on the use case of opening different POI categories to varying degrees (`test_category_combinations_for_dashboard`), which supplies model predictions for our decision-support dashboard. This will start 1,024 jobs per MSA *and* best-fit parameter set. In our original paper, we also ran other types of use case experiments, such as `test_max_capacity_clipping` and `test_retrospective_counterfactuals`; thus, you will see other options for experiments in the code.

5. **Analyzing models and generating figures for paper**. Once models have been run, figures and results in the paper can be reproduced by running `make_figures.ipynb`.

## Files

**covid_constants_and_util.py**: Constants (e.g., file paths) and general utility methods. 

**helper_methods_for_aggregate_data_analysis.py**: Helper methods used in data processing and throughout the analysis. 

**generate_ipf.py**: Generates POI-CBG mobility networks using iterative proportional fitting.

**disease_model.py**: Implements the disease model on the mobility network. 

**model_experiments.py**: Runs models for the experiments described in the paper. 

**model_results.py**: Helper methods to visualize and analyze results from model experiments.

**process_safegraph_data.ipynb**: Processes the raw SafeGraph data. 

**initialize_seir.ipynb**: Estimates initial states per CBG given a simulation start date.

**make_figures.ipynb**: Once the models have been run, reproduces figures and other results in the paper. 

**safegraph_env_v3.yml**: Used to set up the conda environment. 

