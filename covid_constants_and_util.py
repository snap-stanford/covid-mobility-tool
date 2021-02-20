import getpass
import os
import dask
from multiprocessing.pool import ThreadPool
import platform
import numpy as np

# https://stackoverflow.com/a/51954326/9477154
MAX_NUMPY_CORES = 1
print("Setting numpy cores to %i" % MAX_NUMPY_CORES)
os.environ["MKL_NUM_THREADS"] = str(MAX_NUMPY_CORES)  # this keeps numpy from using every available core. We have to do this BEFORE WE import numpy for the first time.
os.environ["NUMEXPR_NUM_THREADS"] = str(MAX_NUMPY_CORES)
os.environ["OMP_NUM_THREADS"] = str(MAX_NUMPY_CORES)
os.environ["NUMEXPR_MAX_THREADS"] = str(MAX_NUMPY_CORES)
dask.config.set(pool=ThreadPool(MAX_NUMPY_CORES))  # This is to make Dask play nicely with the thread limit. See:
# https://stackoverflow.com/questions/39422092/error-with-omp-num-threads-when-using-dask-distributed
# https://stackoverflow.com/questions/40621543/how-to-specify-the-number-of-threads-processes-for-the-default-dask-scheduler
# Using these settings, things seem to be running without incident on Stanford systems (assuming you don't try to run too many jobs at once).
COMPUTER_WE_ARE_RUNNING_ON = platform.node()
RUNNING_CODE_AT_STANFORD = 'stanford' in COMPUTER_WE_ARE_RUNNING_ON.lower()
print("Running code on %s; at Stanford=%s" % (COMPUTER_WE_ARE_RUNNING_ON, RUNNING_CODE_AT_STANFORD))

# common packages needed across files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import time
import math
import random
from dask.diagnostics import ProgressBar
import dask.dataframe as dd

#######################################################################
# DIRECTORIES
#######################################################################
# in old base dir (scratch1)
BASE_DIR = '/dfs/scratch1/safegraph_homes/'
UNZIPPED_DATA_DIR = os.path.join(BASE_DIR, 'all_aggregate_data/20191213-safegraph-aggregate-longitudinal-data-to-unzip-to/')
ANNOTATED_H5_DATA_DIR = os.path.join(BASE_DIR, 'all_aggregate_data/chunks_with_demographic_annotations/')
CHUNK_FILENAME = 'chunk_1.2017-3.2020_c2.h5'
STRATIFIED_BY_AREA_DIR = os.path.join(BASE_DIR, 'all_aggregate_data/chunks_with_demographic_annotations_stratified_by_area/')
PATH_TO_SAFEGRAPH_AREAS = os.path.join(BASE_DIR, 'all_aggregate_data/safegraph_poi_area_calculations/SafeGraphPlacesGeoSupplementSquareFeet.csv.gz')
PATH_TO_IPF_OUTPUT = os.path.join(BASE_DIR, 'all_aggregate_data/ipf_output/')
OLD_FITTED_MODEL_DIR = os.path.join(BASE_DIR, 'all_aggregate_data/fitted_models/')

# in new base dir
NEW_BASE_DIR = '/dfs/scratch2/second_safegraph_homes/'
FITTED_MODEL_DIR = os.path.join(NEW_BASE_DIR, 'extra_safegraph_aggregate_models/')
CURRENT_DATA_DIR = os.path.join(NEW_BASE_DIR, 'all_aggregate_data/raw_safegraph_data/')
NEW_STRATIFIED_BY_AREA_DIR = os.path.join(NEW_BASE_DIR, 'all_aggregate_data/stratified_by_metro_area/')
PATH_TO_NEW_IPF_OUTPUT = os.path.join(NEW_BASE_DIR, 'all_aggregate_data/ipf_output/')
PATH_TO_SEIR_INIT = os.path.join(NEW_BASE_DIR, 'all_aggregate_data/seir_init/')

# supplementary datasets: census, geographical, NYT, Google
PATH_TO_ACS_1YR_DATA = os.path.join(BASE_DIR, 'external_datasets_for_aggregate_analysis/2018_one_year_acs_population_data/nhgis0001_ds239_20185_2018_blck_grp.csv')
PATH_TO_ACS_5YR_DATA = os.path.join(BASE_DIR, 'external_datasets_for_aggregate_analysis/2017_five_year_acs_data/2017_five_year_acs_data.csv')
PATH_TO_CENSUS_BLOCK_GROUP_DATA = os.path.join(BASE_DIR, 'base_dir_for_all_new_data_and_results/non_safegraph_datasets/census_block_group_data/ACS_5_year_2013_to_2017_joined_to_blockgroup_shapefiles/') # census block group boundaries.
PATH_TO_COUNTY_TO_MSA_MAPPING = os.path.join(BASE_DIR, 'base_dir_for_all_new_data_and_results/non_safegraph_datasets/census_block_group_data/august_2017_county_to_metropolitan_mapping.csv') # maps counties to MSAs, consistent with the Experienced Segregation paper. Data was downloaded from https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2017/delineation-files/list1.xls.
PATH_TO_NYT_DATA = os.path.join(BASE_DIR, 'external_datasets_for_aggregate_analysis/nytimes_coronavirus_data/covid-19-data/us-counties.csv')
PATH_TO_OLD_NYT_DATA = os.path.join(BASE_DIR, 'external_datasets_for_aggregate_analysis/nytimes_coronavirus_data/covid-19-data-used-in-nature/us-counties.csv')
PATH_TO_GOOGLE_DATA = os.path.join(BASE_DIR, 'external_datasets_for_aggregate_analysis/20200508_google_mobility_report.csv')
PATH_TO_MASK_USE_DATA = os.path.join(BASE_DIR, 'external_datasets_for_aggregate_analysis/mask-use-ihme')

#######################################################################
# PARAMS USED IN EXPERIMENTS THAT WILL NOT OFTEN CHANGE
#######################################################################
ALL_TIME_PERIODS = ['20181231_20190325', '20190401_20190624', '20190701_20190923', '20190930_20191223',
                    '20191230_20200224', '20200302_20200608', '20200615_20200817', '20200824_20201019',
                    '20201026_20210118', '20210125_20210201']  # POI data saved in chunks of 2-3 months
IPF_ID_PREFIX = '2020-09-01_to_2020-11-30'  # time range to use for loading POI and CBG ids
POI_VISITS_PREFIX = '2018-12-31_to_2021-01-24'  # time range for all saved hourly POI visits

MSAS = ['Atlanta_Sandy_Springs_Roswell_GA',
        'Chicago_Naperville_Elgin_IL_IN_WI',
        'Dallas_Fort_Worth_Arlington_TX',
        'Houston_The_Woodlands_Sugar_Land_TX',
        'Los_Angeles_Long_Beach_Anaheim_CA',
        'Miami_Fort_Lauderdale_West_Palm_Beach_FL',
        'New_York_Newark_Jersey_City_NY_NJ_PA',
        'Philadelphia_Camden_Wilmington_PA_NJ_DE_MD',
        'San_Francisco_Oakland_Hayward_CA',
        'Washington_Arlington_Alexandria_DC_VA_MD_WV']
VA_MSAS = ['Washington_Arlington_Alexandria_DC_VA_MD_WV',
           'Richmond_VA',
           'Virginia_Beach_Norfolk_Newport_News_VA_NC',
'Blacksburg_Christiansburg_Radford_VA+Charlottesville_VA+Harrisonburg_VA+Lynchburg_VA+Roanoke_VA+Staunton_Waynesboro_VA']

P0_SICK_RANGE = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
BETA_PLAUSIBLE_RANGE = (0.7, 1.3)
AREA_CLIPPING_BELOW = 5
AREA_CLIPPING_ABOVE = 95
DWELL_TIME_CLIPPING_ABOVE = 90
HOURLY_VISITS_CLIPPING_ABOVE = 95
SUBCATEGORY_CLIPPING_THRESH = 20
TOPCATEGORY_CLIPPING_THRESH = 20
MAX_MODELS_TO_TAKE_PER_MSA = 100
ACCEPTABLE_LOSS_TOLERANCE = 1.2
WGS_84_CRS = {'init' :'epsg:4326'}
LATENCY_PERIOD_IN_DAYS= 4
INFECTIOUS_PERIOD_IN_DAYS= 3.5
DETECTION_RATE = 0.23
DETECTION_LAG = 9
DEATH_RATE = 0.006
DEATH_LAG = 18
MASK_EFFICACY = 0.5
FIXED_HOLIDAY_DATES = ['01-01',  # New Year's Day
                       '07-01',  # Independence Day
                       '12-24',  # Christmas Eve
                       '12-25',  # Christmas Day
                       '12-31']  # New Year's Eve
LOWER_PERCENTILE = 2.5
UPPER_PERCENTILE = 97.5
INCIDENCE_POP = 100000

#######################################################################
# PARAMS USED IN EXPERIMENTS THAT SHOULD BE CHECKED/CHANGED
# IF NECESSARY BEFORE EVERY EXPERIMENT
#######################################################################
MSAS_IMPLEMENTED_FOR_V2 = ['Richmond_VA']  # the list of MSAs you want to run experiments for right now
CPU_USAGE_THRESHOLD = 70
MEM_USAGE_THRESHOLD = 50
SECONDS_TO_WAIT_AFTER_EXCEEDING_COMP_THRESHOLD = 8
SECONDS_TO_WAIT_BETWEEN_JOBS = 8  # make higher if individual jobs will take up lots of memory / compute
SAVE_MODEL_RESULTS_SEPARATELY = False # if set to True, save a pickle of various model fields. Not doing this for now because it's redundant with the model and takes up room.
USERNAME2COMPUTERS = {'emmap1':['rambo', 'trinity'],
                      'serinac':['rambo']} # , 'furiosa', 'madmax2', 'madmax3']}
MIN_TIMESTRING_TO_LOAD_BEST_FIT_MODELS = '2021_01_19_10_48_52_008132'
MAX_TIMESTRING_TO_LOAD_BEST_FIT_MODELS = '2021_01_19_17_02_36_029096'

#### PARAMETERS FOR FIRST WAVE EXPERIMENTS ####
# MIN_DATETIME = datetime.datetime(2020, 3, 15, 0)
# MAX_DATETIME = datetime.datetime(2020, 5, 14, 23)
# TRAIN_TEST_PARTITION = datetime.datetime(2020, 5, 1)
# # beta_and_psi_plausible_range is output of make_param_plausibility_plot and should be updated whenever you recalibrate R0. These numbers allow R0_base to range from 0.1 - 2 and R0_PSI to range from 1-3 in first week of Mar 2020.
# BETA_AND_PSI_PLAUSIBLE_RANGE = {"min_home_beta": 0.0011934944636759236, 
#                                 "max_home_beta": 0.02386988927351847, 
#                                 "min_poi_psi": 1300.3428379028517,
#                                 "max_poi_psi": 6204.298191231721}

#### PARAMETERS FOR SECOND WAVE EXPERIMENTS ####
MIN_DATETIME = datetime.datetime(2020, 11, 1, 0)
MAX_DATETIME = datetime.datetime(2020, 12, 31, 23)
TRAIN_TEST_PARTITION = datetime.datetime(2020, 12, 18)
# # beta_and_psi_plausible_range is output of make_param_plausibility_plot and should be updated whenever you recalibrate R0. These numbers allow R0_base to range from 0.1 - 2 and R0_PSI to range from 0.1 - 2 in first week of Nov 2020.
BETA_AND_PSI_PLAUSIBLE_RANGE = {"min_home_beta": 0.002820909916243525, 
                                "max_home_beta": 0.056418198324870494, 
                                "min_poi_psi": 628.3147342601472,
                                "max_poi_psi": 20177.930428375937}

#######################################################################
# USEFUL DICTIONARIES / LISTS
#######################################################################
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

codes_to_states = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AS": "American Samoa",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "Washington, D.C.",
    "FM": "Federated States Of Micronesia",
    "FL": "Florida",
    "GA": "Georgia",
    "GU": "Guam",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MH": "Marshall Islands",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "MP": "Northern Mariana Islands",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PW": "Palau",
    "PA": "Pennsylvania",
    "PR": "Puerto Rico",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VI": "Virgin Islands",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming"
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

MSAS_TO_PRETTY_NAMES = {'Atlanta_Sandy_Springs_Roswell_GA':'Atlanta',
                        'Chicago_Naperville_Elgin_IL_IN_WI':"Chicago",
                        'Dallas_Fort_Worth_Arlington_TX':"Dallas",
                        'Houston_The_Woodlands_Sugar_Land_TX':"Houston",
                        'Los_Angeles_Long_Beach_Anaheim_CA':"Los Angeles",
                        'Miami_Fort_Lauderdale_West_Palm_Beach_FL':"Miami",
                        'New_York_Newark_Jersey_City_NY_NJ_PA':"New York City",
                        'Philadelphia_Camden_Wilmington_PA_NJ_DE_MD':"Philadelphia",
                        'San_Francisco_Oakland_Hayward_CA':"San Francisco",
                        'Washington_Arlington_Alexandria_DC_VA_MD_WV':"Washington DC",
                        'Richmond_VA':'Richmond',
                        'Virginia_Beach_Norfolk_Newport_News_VA_NC':'Eastern',
                        'Blacksburg_Christiansburg_Radford_VA+Charlottesville_VA+Harrisonburg_VA+Lynchburg_VA+Roanoke_VA+Staunton_Waynesboro_VA':'Joint VA MSAs'}

MSAS_TO_STATE_CBG_FILES = {'Washington_Arlington_Alexandria_DC_VA_MD_WV':['ACS_2017_5YR_BG_11_DISTRICT_OF_COLUMBIA.gdb',
                                                        'ACS_2017_5YR_BG_24_MARYLAND.gdb',
                                                        'ACS_2017_5YR_BG_51_VIRGINIA.gdb',
                                                        'ACS_2017_5YR_BG_54_WEST_VIRGINIA.gdb'],
                      'Atlanta_Sandy_Springs_Roswell_GA':['ACS_2017_5YR_BG_13_GEORGIA.gdb'],
                      'Chicago_Naperville_Elgin_IL_IN_WI':['ACS_2017_5YR_BG_17_ILLINOIS.gdb',
                                                          'ACS_2017_5YR_BG_18_INDIANA.gdb',
                                                          'ACS_2017_5YR_BG_55_WISCONSIN.gdb'],
                      'Dallas_Fort_Worth_Arlington_TX':['ACS_2017_5YR_BG_48_TEXAS.gdb'],
                      'Houston_The_Woodlands_Sugar_Land_TX':['ACS_2017_5YR_BG_48_TEXAS.gdb'],
                      'Los_Angeles_Long_Beach_Anaheim_CA':['ACS_2017_5YR_BG_06_CALIFORNIA.gdb'],
                      'Miami_Fort_Lauderdale_West_Palm_Beach_FL':['ACS_2017_5YR_BG_12_FLORIDA.gdb'],
                      'New_York_Newark_Jersey_City_NY_NJ_PA':['ACS_2017_5YR_BG_36_NEW_YORK.gdb',
                                                              'ACS_2017_5YR_BG_34_NEW_JERSEY.gdb',
                                                              'ACS_2017_5YR_BG_42_PENNSYLVANIA.gdb'],
                      'Philadelphia_Camden_Wilmington_PA_NJ_DE_MD':['ACS_2017_5YR_BG_42_PENNSYLVANIA.gdb',
                      'ACS_2017_5YR_BG_34_NEW_JERSEY.gdb',
                      'ACS_2017_5YR_BG_24_MARYLAND.gdb',
                      'ACS_2017_5YR_BG_10_DELAWARE.gdb'],
                      'San_Francisco_Oakland_Hayward_CA':['ACS_2017_5YR_BG_06_CALIFORNIA.gdb']}

# in analysis, we remove same categories as MIT sloan paper, or try to. They write:
# We omit “Bars and Clubs” as SafeGraph seems to dramatically undercount these locations. We omit “Parks and Playgrounds” as SafeGraph struggles to precisely define the bor- ders of these irregularly shaped points of interest. We omit “Public and Private Schools” and “Child Care and Daycare Centers” due to challenges in adjusting for the fact that individuals under the age of 13 are not well tracked by SafeGraph.
REMOVED_SUBCATEGORIES = ['Child Day Care Services',
'Elementary and Secondary Schools',
'Drinking Places (Alcoholic Beverages)',
'Nature Parks and Other Similar Institutions',
'General Medical and Surgical Hospitals',
'Malls', # parent ID problem
'Colleges, Universities, and Professional Schools', # parent ID problem
'Amusement and Theme Parks', # parent ID problem
'Other Airport Operations']

SUBCATEGORIES_TO_PRETTY_NAMES = {
    'Golf Courses and Country Clubs':'Golf Courses & Country Clubs',
    'Other Gasoline Stations':'Other Gas Stations',
    'Malls':'Malls',
    'Gasoline Stations with Convenience Stores':'Gas Stations',
    'New Car Dealers':'New Car Dealers',
    'Pharmacies and Drug Stores':'Pharmacies & Drug Stores',
    'Department Stores':'Department Stores',
    'Convenience Stores':'Convenience Stores',
    'All Other General Merchandise Stores':'Other General Stores',
    'Nature Parks and Other Similar Institutions':'Parks & Similar Institutions',
    'Automotive Parts and Accessories Stores':'Automotive Parts Stores',
    'Supermarkets and Other Grocery (except Convenience) Stores':'Grocery Stores',
    'Pet and Pet Supplies Stores':'Pet Stores',
    'Used Merchandise Stores':'Used Merchandise Stores',
    'Sporting Goods Stores':'Sporting Goods Stores',
    'Beer, Wine, and Liquor Stores':'Liquor Stores',
    'Insurance Agencies and Brokerages':'Insurance Agencies',
    'Gift, Novelty, and Souvenir Stores':'Gift Stores',
    'General Automotive Repair':'Car Repair Shops',
    'Limited-Service Restaurants':'Limited-Service Restaurants',
    'Snack and Nonalcoholic Beverage Bars':'Cafes & Snack Bars',
    'Offices of Physicians (except Mental Health Specialists)':'Offices of Physicians',
    'Fitness and Recreational Sports Centers':'Fitness Centers',
    'Musical Instrument and Supplies Stores':'Musical Instrument Stores',
    'Full-Service Restaurants':'Full-Service Restaurants',
    'Insurance Agencies':'Insurance Agencies',
    'Hotels (except Casino Hotels) and Motels':'Hotels & Motels',
    'Hardware Stores':'Hardware Stores',
    'Religious Organizations':'Religious Organizations',
    'Offices of Dentists':'Offices of Dentists',
    'Home Health Care Services':'Home Health Care Services',
    'Used Merchandise Stores':'Used Merchandise Stores',
    'General Medical and Surgical Hospitals':'General Hospitals',
    'Colleges, Universities, and Professional Schools':'Colleges & Universities',
    'Commercial Banking':'Commercial Banking',
    'Used Car Dealers':'Used Car Dealers',
    'Hobby, Toy, and Game Stores':'Hobby & Toy Stores',
    'Other Airport Operations':'Other Airport Operations',
    'Optical Goods Stores':'Optical Goods Stores',
    'Electronics Stores':'Electronics Stores',
    'Tobacco Stores':'Tobacco Stores',
    'All Other Amusement and Recreation Industries':'Other Recreation Industries',
    'Book Stores':'Book Stores',
    'Office Supplies and Stationery Stores':'Office Supplies',
    'Drinking Places (Alcoholic Beverages)':'Bars (Alc. Beverages)',
    'Furniture Stores':'Furniture Stores',
    'Assisted Living Facilities for the Elderly':'Senior Homes',
    'Sewing, Needlework, and Piece Goods Stores':'Sewing & Piece Goods Stores',
    'Cosmetics, Beauty Supplies, and Perfume Stores':'Cosmetics & Beauty Stores',
    'Amusement and Theme Parks':'Amusement & Theme Parks',
    'All Other Home Furnishings Stores':'Other Home Furnishings Stores',
    'Offices of Mental Health Practitioners (except Physicians)':'Offices of Mental Health Practitioners',
    'Carpet and Upholstery Cleaning Services':'Carpet Cleaning Services',
    'Florists':'Florists',
    'Women\'s Clothing Stores':'Women\'s Clothing Stores',
    'Family Clothing Stores':'Family Clothing Stores',
    'Jewelry Stores':'Jewelry Stores',
    'Beauty Salons':'Beauty Salons',
    'Motion Picture Theaters (except Drive-Ins)':'Movie Theaters',
    'Libraries and Archives':'Libraries & Archives',
    'Bowling Centers':'Bowling Centers',
    'Casinos (except Casino Hotels)':'Casinos',
    'All Other Miscellaneous Store Retailers (except Tobacco Stores)':'Other Misc. Retail Stores',
    'RV (Recreational Vehicle) Parks and Campgrounds':'RV Parks amd Campgrounds',
}

CATEGORY_GROUPS = {'Restaurants': ['Full-Service Restaurants', 
                                   'Limited-Service Restaurants', 
                                   'Snack and Nonalcoholic Beverage Bars', 
                                   'Drinking Places (Alcoholic Beverages)'],
                   'Essential Retail': ['Supermarkets and Other Grocery (except Convenience) Stores', 
                                        'Convenience Stores', 
                                        'Pharmacies and Drug Stores'],
                   'Gyms': ['Fitness and Recreational Sports Centers'],
                   'Religious Organizations': ['Religious Organizations'],
                   'Retail': ['Automotive Parts and Accessories Stores',  # every store with at least 50 POIs in DC MSA
                             'Beer, Wine, and Liquor Stores',
                             'Electronics Stores',
                             "Women's Clothing Stores",
                             'Pet and Pet Supplies Stores',
                             'Optical Goods Stores',
                             'Used Merchandise Stores',
                             'All Other Home Furnishings Stores',
                             'Furniture Stores',
                             'Sporting Goods Stores',
                             'All Other General Merchandise Stores',
                             'Jewelry Stores',
                             'Family Clothing Stores',
                             'Cosmetics, Beauty Supplies, and Perfume Stores',
                             'Hardware Stores',
                             'Shoe Stores',
                             'Tobacco Stores',
                             'Gift, Novelty, and Souvenir Stores',
                             'Department Stores',
                             'Musical Instrument and Supplies Stores',
                             'Office Supplies and Stationery Stores',
                             'Nursery, Garden Center, and Farm Supply Stores',
                             'Sewing, Needlework, and Piece Goods Stores',
                             'Hobby, Toy, and Game Stores',
                             'Book Stores',
                             'Food (Health) Supplement Stores',
                             'Paint and Wallpaper Stores',
                             "Men's Clothing Stores",
                             'Other Clothing Stores',
                             'Floor Covering Stores',
                             'All Other Miscellaneous Store Retailers (except Tobacco Stores)',
                             'Confectionery and Nut Stores',
                             "Children's and Infants' Clothing Stores",
                             'Luggage and Leather Goods Stores']}

def get_datetime_hour_as_string(datetime_hour):
    return '%i.%i.%i.%i' % (datetime_hour.year, datetime_hour.month,
                            datetime_hour.day, datetime_hour.hour)

def load_csv_possibly_with_dask(filenames, use_dask=False, compression='gzip', blocksize=None, compute_with_dask=True, **kwargs):
    # Avoid loading the index column because it's probably not desired.
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
        # Use tqdm to display a progress bar.
        return pd.concat(pd.read_csv(f, **kwargs) for f in tqdm_wrap(filenames))

def get_cumulative(x):
    '''
    Converts an array of values into its cumulative form,
    i.e. cumulative_x[i] = x[0] + x[1] + ... + x[i]

    x should either be a 1D or 2D numpy array. If x is 2D,
    the cumulative form of each row is returned.
    '''
    assert len(x.shape) in [1, 2]
    if len(x.shape) == 1:
        cumulative_x = []
        curr_sum = 0
        for val in x:
            curr_sum = curr_sum + val
            cumulative_x.append(curr_sum)
        cumulative_x = np.array(cumulative_x)
    else:
        num_seeds, num_time = x.shape
        cumulative_x = []
        curr_sum = np.zeros(num_seeds)
        for i in range(num_time):
            curr_sum = curr_sum + x[:, i]
            cumulative_x.append(curr_sum.copy())
        cumulative_x = np.array(cumulative_x).T
    return cumulative_x

def get_daily_from_cumulative(x):
    '''
    Converts an array of values from its cumulative form
    back into its original form.

    x should either be a 1D or 2D numpy array.
    '''
    assert len(x.shape) in [1, 2]
    if len(x.shape) == 1:
        arr_to_return = np.array([x[0]] + list(x[1:] - x[:-1]))
    else:
        # seeds are axis 0, so want to subtract along axis 1.
        x0 = x[:, :1]
        increments = x[:, 1:] - x[:, :-1]
        arr_to_return = np.concatenate((x0, increments), axis=1)
    if not (arr_to_return >= 0).all():
        bad_val_frac = (arr_to_return < 0).mean()
        if bad_val_frac > 0.1:
            print("Warning: fraction %2.3f of values are not greater than 0! clipping to 0" % bad_val_frac)
#         assert bad_val_frac < 0.1 # this happens quite occasionally in NYT data.
        arr_to_return = np.clip(arr_to_return, 0, None)
    return arr_to_return

def mean_and_CIs_of_timeseries_matrix(M, alpha=0.05):
    """
    Given a matrix which is N_SEEDS X T, return mean and upper and lower CI for plotting.
    """
    assert alpha > 0
    assert alpha < 1
    mean = np.mean(M, axis=0)
    lower_CI = np.percentile(M, 100 * alpha/2, axis=0)
    upper_CI = np.percentile(M, 100 * (1 - alpha/2), axis=0)
    return mean, lower_CI, upper_CI

def apply_smoothing(x, agg_func=np.mean, before=2, after=2):
    new_x = []
    for i, x_point in enumerate(x):
        before_idx = max(0, i-before)
        after_idx = min(len(x), i+after+1)
        new_x.append(agg_func(x[before_idx:after_idx]))
    return np.array(new_x)

# inspired by https://dfrieds.com/data-visualizations/how-format-large-tick-values.html
def reformat_large_tick_values(tick_val, pos):
    if tick_val >= 1000000000:
        val = round(tick_val/1000000000, 1)
        postfix = 'B'
    elif tick_val >= 10000000:  # if over 10M, don't include decimal
        val = int(round(tick_val/1000000, 0))
        postfix = 'M'
    elif tick_val >= 1000000:  # if 1M-10M, include decimal
        val = round(tick_val/1000000, 1)
        postfix = 'M'
    elif tick_val >= 1000:
        val = int(round(tick_val/1000, 0))
        postfix = 'k'
    else:
        val = int(tick_val)
        postfix = ''
    new_tick_format = '%s%s' % (val, postfix)
    return new_tick_format

def reformat_decimal_as_percent(tick_val, pos):
    percent = round(tick_val * 100, 1)
    new_tick_format = '%d%%' % percent
    return new_tick_format
