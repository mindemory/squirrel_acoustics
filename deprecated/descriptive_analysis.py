import os
import numpy as np
import pandas as pd

from params import *
from plots import note_distribution, inter_note_distribution
from stats_plotter import note_stats_master, note_stats_file

df_path = os.path.join(PROJECT_PATH, 'dataframes')
# Load dataframes
print("Loading dataframes")
master_df = pd.read_csv(os.path.join(df_path, 'master_df.csv'))
master_good_df = pd.read_csv(os.path.join(df_path, 'master_good_df.csv'))
master_random_df = pd.read_csv(os.path.join(df_path, 'master_random_df.csv'))
file_df = pd.read_csv(os.path.join(df_path, 'file_df.csv'))
file_good_df = pd.read_csv(os.path.join(df_path, 'file_good_df.csv'))

cols_to_consider = ['Species', 'Location', 'Median Low Freq (Hz)', 'Median High Freq (Hz)',
       'Median Delta Freq (Hz)', 'Median Delta Time (s)',
       'Median Inter_note_difference (s)', 'Note density (notes per s)',
       'Sub-bout density (sub-bouts per s)', 'Bout density (bouts per s)',
       'Unique note count']
cols_to_consider_blurb = ['Median Low Freq (Hz)', 'Median High Freq (Hz)',
       'Median Delta Freq (Hz)', 'Median Delta Time (s)',
       'Median Inter_note_difference (s)', 'Note density (notes per s)',
       'Sub-bout density (sub-bouts per s)', 'Bout density (bouts per s)',
       'Unique note count']
#file_random_df = pd.read_csv(os.path.join(df_path, 'file_random_df.csv'))
#print(file_good_df.columns)
selected_features_notes = ['PFC Max Freq (Hz)', 'PFC Min Freq (Hz)', 'BW 90% (Hz)']
selected_features_files = ['Median Delta Freq (Hz)', 'Median High Freq (Hz)', 'Stdev High Freq (Hz)']

#df_folder = os.path.join(PROJECT_PATH, 'descriptive_stats')
df_folder = os.path.join(PROJECT_PATH, 'Figures')
if not os.path.exists(df_folder):
    os.mkdir(df_folder)

note_stats_master(master_good_df, df_folder)
note_stats_file(file_good_df, df_folder, cols_to_consider, cols_to_consider_blurb)

#print("Running preliminary analysis")
#print("Saving note distribution")
#note_distribution(master_good_df, species_list)
#master_df_bout_version = pd.DataFrame()
#for bf in master_df['File_name'].unique():
#    temp_df = master_df[master_df['File_name'] == bf].reset_index(drop = True)
#    temp_df = temp_df.drop([0]).reset_index(drop = True)
#    master_df_bout_version = pd.concat([master_df_bout_version, temp_df])

# Code section for computing the bout thresholds. Uncomment to run
# for spp in master_df_bout_version['Species'].unique():
#     bout_df_temp = master_df_bout_version[master_df_bout_version['Species'] == spp]
#     print(bout_df_temp['Inter_note_difference (s)'].min())
#     bout_df_temp = bout_df_temp[bout_df_temp['Inter_note_difference (s)'] > sub_bout_difference_dict_F[spp]]
#     print(bout_df_temp['Inter_note_difference (s)'].min())
#     print(spp)
#     Q1, M, Q3 = bout_df_temp[bout_df_temp['Species'] == spp]['Inter_note_difference (s)'].quantile([0.25, 0.5, 0.75])
#     IQR = Q3 - Q1
#     LB = Q1 - 1.5 * IQR
#     UB = Q3 + 1.5 * IQR
#     print(Q1, M, Q3)
#     print(IQR)
#     print(LB, UB)
#     print()
#print("Saving internote distribution")
#inter_note_distribution(species_list, master_df_bout_version)
