# Loading packages
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Loading files and functions
import pre_proc_func as ppf
from params import *
from plots import note_distribution, inter_note_distribution

PROJECT_PATH = os.path.join('/home/mrugank/Sciurid Lab/codes/july_analysis')

# Check the files for any errors
print("Screening files for errors")
#ppf.screen_tables(PROJECT_PATH, species_list)
print()

# Run saturation analysis to determine note threshold
#sat_analysis_confirmation = input("""The program will now run Saturation Analysis which can help determine the threshold
#note count for analysis. This can take upto 15 minutes to run and will require you to
#create a "Figures" folder at PROJECT_PATH.
#Do you want to run saturation analysis? (y or n): """)
sat_analysis_confirmation = 'n'
if sat_analysis_confirmation == 'y':
    print("Saturation Analysis is running")
    metric = 'High Freq (Hz)'
    ppf.saturation_analysis(PROJECT_PATH, metric, species_list, ci = 95)
else:
    print("Saturation Analysis has been skipped")
    pass
print()

# Load Location File
location_path = os.path.join(PROJECT_PATH, 'locations.csv')
location_df = pd.read_csv(location_path)
# Add extension to file names wherever needed
for i in range(location_df.shape[0]):
    #print(location_df.loc[i, '12_Audio_file_name'])
    if location_df.loc[i, '12_Audio_file_name'][-4:] != '.wav' and location_df.loc[i, '12_Audio_file_name'][-4:] != '.WAV':
        location_df.loc[i, '12_Audio_file_name'] = location_df.loc[i, '12_Audio_file_name'] + '.wav'

# Create master_df by concatenating all files
print('Creating master_df')
master_df = ppf.generate_master_df(PROJECT_PATH, species_list, location_df)
#file_df = master_df.groupby(['Species', 'File_name']).median()
##print(file_df.head())
#print(file_df.shape)
#print(file_df.columns)

note_distribution(PROJECT_PATH, master_df, species_list)

master_df_bout_version = pd.DataFrame()
for bf in master_df['File_name'].unique():
  temp_df = master_df[master_df['File_name'] == bf].reset_index(drop = True)
  temp_df = temp_df.drop([0]).reset_index(drop = True)
  master_df_bout_version = pd.concat([master_df_bout_version, temp_df])
print(master_df.shape, master_df_bout_version.shape)
inter_note_distribution(PROJECT_PATH, species_list, master_df_bout_version)
