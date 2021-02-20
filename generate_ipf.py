from covid_constants_and_util import *
from model_experiments import *
from model_results import *
import helper_methods_for_aggregate_data_analysis as helper
import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
import datetime
import copy
import dask
import re
import scipy 
import time 

TIME_PERIODS_FOR_CURR_IPF = ['20210125_20210201']
MIN_DATETIME_FOR_CURR_IPF = datetime.datetime(2021, 1, 25, 0)
MAX_DATETIME_FOR_CURR_IPF = datetime.datetime(2021, 2, 7, 23)
TEST_IPF_HOUR = datetime.datetime(2020, 10, 1, 0)
CBG_COUNT_CUTOFF = 100
POI_HOURLY_VISITS_CUTOFF = 'all'

def generate_ipf_for_msa(msa_name, merged_df, min_datetime, max_datetime, poi_ids=None, cbg_ids=None,
                         update_poi_hourly_visits=True):
    m = fit_disease_model_on_real_data(d=merged_df,
                                   min_datetime=min_datetime,
                                   max_datetime=max_datetime,
                                   msa_name=msa_name,
                                   exogenous_model_kwargs={'poi_psi':1, 
                                                           'home_beta':1, 
                                                           'p_sick_at_t0':0,  # don't need infections
                                                           'just_compute_r0':False},
                                   poi_attributes_to_clip={'clip_areas':True, 
                                                           'clip_dwell_times':True, 
                                                           'clip_visits':True},
                                   preload_poi_visits_list_filename=None,
                                   poi_cbg_visits_list=None,
                                   poi_ids=poi_ids,
                                   cbg_ids=cbg_ids,
                                   correct_poi_visits=True,
                                   multiply_poi_visit_counts_by_census_ratio=True,
                                   aggregate_home_cbg_col='aggregated_cbg_population_adjusted_visitor_home_cbgs',
                                   poi_hourly_visits_cutoff=POI_HOURLY_VISITS_CUTOFF,  
                                   cbg_count_cutoff=CBG_COUNT_CUTOFF,
                                   cbgs_to_filter_for=None,
                                   cbg_groups_to_track=None,
                                   counties_to_track=None,
                                   include_cbg_prop_out=True,
                                   include_inter_cbg_travel=False,
                                   include_mask_use=False,
                                   model_init_kwargs={'ipf_final_match':'poi',
                                                      'ipf_num_iter':100,
                                                      'num_seeds':2},
                                   simulation_kwargs={'do_ipf':True, 
                                                      'allow_early_stopping':False},
                                   counterfactual_poi_opening_experiment_kwargs=None,
                                   counterfactual_retrospective_experiment_kwargs=None,
                                   return_model_without_fitting=False,
                                   attach_data_to_model=True,
                                   model_quality_dict=None,
                                   verbose=True)
                                   
    ipf_out = m.poi_cbg_visit_history
    poi_labels = m.d.index
    cbg_labels = m.ALL_UNIQUE_CBGS
    assert ipf_out[0].shape == (len(poi_labels), len(cbg_labels))
    
    if poi_ids is None:  # don't resave if POI IDs were given
        poi_df = pd.DataFrame({'poi_id':poi_labels})
        path_to_poi_csv = os.path.join(PATH_TO_NEW_IPF_OUTPUT, '%s/%s_to_%s_poi_ids.csv' % 
                                   (msa_name, min_datetime.strftime('%Y-%m-%d'), max_datetime.strftime('%Y-%m-%d')))
        poi_df.to_csv(path_to_poi_csv, index=False)
        print(path_to_poi_csv)
    
    if cbg_ids is None:  # don't resave if CBG IDs were given
        cbg_df = pd.DataFrame({'cbg_id':cbg_labels})
        path_to_cbg_csv = os.path.join(PATH_TO_NEW_IPF_OUTPUT, '%s/%s_to_%s_cbg_ids.csv' % 
                                       (msa_name, min_datetime.strftime('%Y-%m-%d'), max_datetime.strftime('%Y-%m-%d')))
        cbg_df.to_csv(path_to_cbg_csv, index=False)
        print(path_to_cbg_csv)
    
    all_hours = helper.list_hours_in_range(min_datetime, max_datetime)
    assert len(ipf_out) == len(all_hours)
    for i, (hr, mat) in enumerate(list(zip(all_hours, ipf_out))):
        path_to_npz = os.path.join(PATH_TO_NEW_IPF_OUTPUT, '%s/%s.npz' % (msa_name, hr.strftime('%Y-%m-%d-%H')))
        if i % 168 == 0:
            print(path_to_npz)
        scipy.sparse.save_npz(path_to_npz, mat)   
    
    if update_poi_hourly_visits:
        print('Updating POI hourly visits .npy matrix...')
        append_new_ipf_hours_to_hourly_visits_per_poi(msa_name, min_datetime, max_datetime, ipf_out=ipf_out)
    return ipf_out, poi_labels, cbg_labels

def append_new_ipf_hours_to_hourly_visits_per_poi(msa_name, min_datetime, max_datetime, ipf_out=None):
    min_hour, max_hour = POI_VISITS_PREFIX.split('_to_')
    year, month, date = min_hour.split('-')
    orig_min_datetime = datetime.datetime(int(year), int(month), int(date), 0)
    year, month, date = max_hour.split('-')
    orig_max_datetime = datetime.datetime(int(year), int(month), int(date), 23)
    print('Original hour range: %s to %s' % 
          (orig_min_datetime.strftime('%Y-%m-%d, %H'), orig_max_datetime.strftime('%Y-%m-%d, %H')))
    print('New hour range: %s to %s' % 
          (min_datetime.strftime('%Y-%m-%d, %H'), max_datetime.strftime('%Y-%m-%d, %H')))
    # there should be no gaps between original range and new range
    orig_hours = helper.list_hours_in_range(orig_min_datetime, orig_max_datetime)
    if min_datetime <= orig_max_datetime:
        print('Warning: found overlapping hours btwn new IPF output and saved hourly visits -> using row sums from new IPF output')
        orig_end_idx = orig_hours.index(min_datetime)  # truncate old data and use new data for repeated hours
    elif min_datetime == (orig_max_datetime + datetime.timedelta(hours=1)):  # new data directly follows old data
        print('New IPF data directly follows old IPF data')
        orig_end_idx = len(orig_hours)
    else:
        raise Exception('There is a gap btwn new IPF output and saved hourly visits')
    
    orig_path = os.path.join(PATH_TO_NEW_IPF_OUTPUT, '%s/%s_poi_hourly_visits.npy' % (msa_name, POI_VISITS_PREFIX))
    orig_poi_visits = np.load(orig_path)
    if ipf_out is None:
        ipf_out = helper.load_ipf_output_for_hours(msa_name, min_datetime, max_datetime, return_ids=False)
    new_poi_visits = np.array([m @ np.ones(m.shape[1]) for m in ipf_out]).T  # sum over CBGs per POI x hour
    combined_poi_visits = np.concatenate([orig_poi_visits[:, :orig_end_idx], new_poi_visits], axis=1)
    combined_hours = helper.list_hours_in_range(orig_min_datetime, max_datetime)
    assert combined_poi_visits.shape == (orig_poi_visits.shape[0], len(combined_hours))  # num_pois x num_hours
    combined_prefix = '%s_to_%s' % (combined_hours[0].strftime('%Y-%m-%d'), combined_hours[-1].strftime('%Y-%m-%d'))
    combined_path = os.path.join(PATH_TO_NEW_IPF_OUTPUT, '%s/%s_poi_hourly_visits.npy' % (msa_name, combined_prefix))
    print('Saving at', combined_path)
    ts = time.time()
    np.save(combined_path, combined_poi_visits)
    duration = time.time() - ts
    print('Time to save = %.2fs. Size of .npy = %.2fMB' % (duration, os.path.getsize(combined_path) / (1024**2)))
    
def generate_hourly_visits_per_poi(msa_name, msa_df_for_experiments, min_datetime, max_datetime, test_ipf_hour):
    all_hours = helper.list_hours_in_range(min_datetime, max_datetime)
    assert test_ipf_hour in all_hours
    ipf_hour_index = all_hours.index(test_ipf_hour)
    poi_ids = helper.load_poi_ids_for_msa(msa_name)
    cbg_ids = helper.load_cbg_ids_for_msa(msa_name)
    print('Loaded %d POIs and %d CBGs' % (len(poi_ids), len(cbg_ids)))
    path_to_npz = os.path.join(PATH_TO_NEW_IPF_OUTPUT, '%s/%s.npz' % (msa_name, test_ipf_hour.strftime('%Y-%m-%d-%H')))
    test_ipf_out = scipy.sparse.load_npz(path_to_npz)
    assert test_ipf_out.shape == (len(poi_ids), len(cbg_ids))
    
    m = fit_disease_model_on_real_data(d=msa_df_for_experiments,
                                   min_datetime=min_datetime,
                                   max_datetime=max_datetime,
                                   msa_name=msa_name,
                                   exogenous_model_kwargs={'poi_psi':1, 
                                                           'home_beta':1, 
                                                           'p_sick_at_t0':0, 
                                                           'just_compute_r0':False},
                                   poi_attributes_to_clip={'clip_areas':True, 
                                                           'clip_dwell_times':True, 
                                                           'clip_visits':True},
                                   preload_poi_visits_list_filename=None,
                                   poi_cbg_visits_list=None,
                                   poi_ids=poi_ids,
                                   cbg_ids=cbg_ids,
                                   correct_poi_visits=True,
                                   multiply_poi_visit_counts_by_census_ratio=True,
                                   aggregate_home_cbg_col='aggregated_cbg_population_adjusted_visitor_home_cbgs',
                                   poi_hourly_visits_cutoff='all',  
                                   cbg_count_cutoff=100,
                                   cbgs_to_filter_for=None,
                                   cbg_groups_to_track=None,
                                   counties_to_track=None,
                                   include_cbg_prop_out=False,
                                   include_inter_cbg_travel=False,
                                   include_mask_use=False,
                                   model_init_kwargs={'ipf_final_match':'poi',
                                                      'ipf_num_iter':100,
                                                      'num_seeds':2},
                                   simulation_kwargs={'do_ipf':True, 
                                                      'allow_early_stopping':False},
                                   counterfactual_poi_opening_experiment_kwargs=None,
                                   counterfactual_retrospective_experiment_kwargs=None,
                                   return_model_without_fitting=True,
                                   attach_data_to_model=True,
                                   model_quality_dict=None,
                                   verbose=True)
    
    poi_hourly_visits = m.POI_TIME_COUNTS
    assert poi_hourly_visits.shape == (len(poi_ids), len(all_hours))
    poi_props_represented = m.POI_CBG_PROPORTIONS @ np.ones(len(cbg_ids))  # what proportion of each POI's visitors is modeled
    scaled_poi_hourly_visits = (poi_hourly_visits.T * poi_props_represented).T
    target_row_sums = scaled_poi_hourly_visits[:, ipf_hour_index]
    ipf_row_sums = test_ipf_out @ np.ones(len(cbg_ids))
    total_diff = np.sum(np.absolute(ipf_row_sums - target_row_sums))
    assert np.isclose(total_diff, 0)  # make sure hourly visits here matches IPF output
    
    save_fn = '%s_to_%s_poi_hourly_visits.npy' % (min_datetime.strftime('%Y-%m-%d'), max_datetime.strftime('%Y-%m-%d'))
    full_path = os.path.join(PATH_TO_NEW_IPF_OUTPUT, '%s/%s' % (msa_name, save_fn))
    print('Saving at', full_path)
    ts = time.time()
    np.save(full_path, scaled_poi_hourly_visits)
    duration = time.time() - ts
    print('Time to save = %.2fs. Size of .npy = %.2fMB' % (duration, os.path.getsize(full_path) / (1024**2)))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('msa_names', type=str)
    parser.add_argument('--process', type=str, default='ipf', choices=['ipf', 'poi_visits'])
    parser.add_argument('--use_existing_ids', type=bool, default=False)
    parser.add_argument('--outer_or_inner_call', type=str, default='outer', choices=['outer', 'inner'])
    parser.add_argument('--update_poi_visits', type=bool, default=True)
    args = parser.parse_args()
    msa_names = args.msa_names.split(',')
    joint_msa_key = '+'.join(msa_names)
    msa_ipf_dir = os.path.join(PATH_TO_NEW_IPF_OUTPUT, joint_msa_key)
    print('Will save IPF results in', msa_ipf_dir)
    if not os.path.isdir(msa_ipf_dir):
        assert args.process == 'ipf'  # this shouldn't happen if we are on the poi visits step
        print('Making directory')
        cmd = 'mkdir %s' % msa_ipf_dir
        os.system(cmd)
        
    outer_or_inner_call = args.outer_or_inner_call
    if outer_or_inner_call == 'outer':  # redirect outer call to inner call with logfile
        if args.process == 'ipf':
            logfile = os.path.join(msa_ipf_dir, '%s_to_%s_log.out' % 
                          (MIN_DATETIME_FOR_CURR_IPF.strftime('%Y-%m-%d'), MAX_DATETIME_FOR_CURR_IPF.strftime('%Y-%m-%d')))
        else:
            logfile = os.path.join(msa_ipf_dir, 'poi_visits_log.out')
        cmd = 'nohup python -u generate_ipf.py %s --process %s --use_existing_ids %s --update_poi_visits %s --outer_or_inner_call inner > %s 2>&1 &' % (args.msa_names, args.process, args.use_existing_ids, args.update_poi_visits, logfile)
        print("Inner command: %s" % cmd)
        os.system(cmd)
    
    else:
        if args.process == 'ipf':
            if args.use_existing_ids:
                poi_ids = helper.load_poi_ids_for_msa(joint_msa_key)
                cbg_ids = helper.load_cbg_ids_for_msa(joint_msa_key)
                print('Loaded %d POI and %d CBG ids' % (len(poi_ids), len(cbg_ids)))
            else:
                poi_ids = None
                cbg_ids = None
            msa_df = helper.prep_msa_df_for_model_experiments(joint_msa_key, TIME_PERIODS_FOR_CURR_IPF)
            out = generate_ipf_for_msa(joint_msa_key, msa_df, MIN_DATETIME_FOR_CURR_IPF, MAX_DATETIME_FOR_CURR_IPF,
                                       poi_ids=poi_ids, cbg_ids=cbg_ids, update_poi_hourly_visits=args.update_poi_visits)
        else:
            msa_df = helper.prep_msa_df_for_model_experiments(joint_msa_key, ALL_TIME_PERIODS)
            # we want min and max hours; maximum range possible
            hourly_visit_cols = [col for col in msa_df.columns if col.startswith('hourly_visits_')]
            min_date = hourly_visit_cols[0].split('_')[-1]
            year, month, date, hour = min_date.split('.')
            min_datetime = datetime.datetime(int(year), int(month), int(date), int(hour))
            max_date = hourly_visit_cols[-1].split('_')[-1]
            year, month, date, hour = max_date.split('.')
            max_datetime = datetime.datetime(int(year), int(month), int(date), int(hour))
            generate_hourly_visits_per_poi(joint_msa_key, msa_df, min_datetime, max_datetime, TEST_IPF_HOUR)
