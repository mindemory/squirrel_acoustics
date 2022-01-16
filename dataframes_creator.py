# Loading packages
import os
import numpy as np
import pandas as pd

# Loading files and functions
import pre_proc_func as ppf
from params import *

# Check the files for any errors
print("Screening files for errors")
ppf.screen_tables(species_list)
print()

# Run saturation analysis to determine note threshold
sat_analysis_confirmation = input("""The program will now run Saturation Analysis which can help determine the threshold
note count for analysis. This can take upto 15 minutes to run and will require you to
create a "Figures" folder at PROJECT_PATH.
Do you want to run saturation analysis? (y or n): """)
if sat_analysis_confirmation == 'y':
    print("Saturation Analysis is running")
    metric = 'PFC Max Freq (Hz)'
    ppf.saturation_analysis(metric, species_list, ci = 95)
else:
    print("Saturation Analysis has been skipped")
    pass
print()

# Load Location File
location_path = os.path.join(PROJECT_PATH, 'locations.csv')
location_df = pd.read_csv(location_path)

# Add extension to file names wherever needed
for i in range(location_df.shape[0]):
    if location_df.loc[i, '12_Audio_file_name'][-4:] != '.wav' and location_df.loc[i, '12_Audio_file_name'][-4:] != '.WAV':
        #print(location_df.loc[i, '12_Audio_file_name'])
        location_df.loc[i, '12_Audio_file_name'] = location_df.loc[i, '12_Audio_file_name'] + '.wav'

# Create master_df by concatenating all files
df_folder = os.path.join(PROJECT_PATH, 'dataframes')
if not os.path.exists(df_folder):
    os.mkdir(df_folder)

print('Creating master_df')
master_df = ppf.generate_master_df(species_list, location_df)
master_good_df = master_df[master_df['Quality'] == 'G']

master_df.to_csv(os.path.join(df_folder, 'master_df.csv'), index = False)
master_good_df.to_csv(os.path.join(df_folder, 'master_good_df.csv'), index = False)

# Create a summary_df for all the files
print()
print('Creating summary_df')
df_by_species = master_df.groupby('Species')
df_by_species_good = master_df[master_df['Quality'] == 'G'].groupby('Species')
summary_df_1 = df_by_species.agg({'File_name': 'nunique', 'Note': ['count', 'nunique']})
summary_df_1.columns = ['_'.join(col) for col in summary_df_1.columns]
summary_df_2 = df_by_species_good.agg({'Note': ['count','nunique']})
summary_df_2.columns = ['_good_'.join(col) for col in summary_df_2.columns]
summary_df = summary_df_1.join(summary_df_2)
summary_df.to_csv(os.path.join(df_folder, 'summary_df.csv'))

# Create a dataframe with random sampling of notes from each file
print()
print('Creating randomly sampled dataframe')
master_random_df = pd.DataFrame()
beg_file = master_good_df['File_name'].unique()
for bf in beg_file:
    sp_df = master_good_df[master_good_df['File_name'] == bf]
    threshold_value = sample_threshold_dict[sp_df['Species'].unique()[0]]
    #print(threshold_dict[sp_df['species'].unique()[0]])
    if sp_df.shape[0] > threshold_value:
        sp_df = sp_df.sample(threshold_value, random_state = 42)
    master_random_df = pd.concat([master_random_df, sp_df], ignore_index = True)
master_random_df.to_csv(os.path.join(df_folder, 'master_random_df.csv'), index = False)

file_df = ppf.generate_file_df(master_df)
file_good_df = ppf.generate_file_df(master_good_df)
file_random_df = ppf.generate_file_df(master_random_df)

file_df.to_csv(os.path.join(df_folder, 'file_df.csv'), index = False)
file_good_df.to_csv(os.path.join(df_folder, 'file_good_df.csv'), index = False)
file_random_df.to_csv(os.path.join(df_folder, 'file_random_df.csv'), index = False)
# print(file_df.head())
# print(file_df.shape)
# print(file_df.columns)
# print(file_df.isna().sum())
