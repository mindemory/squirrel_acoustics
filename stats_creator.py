# Loading packages
import os
import numpy as np
import pandas as pd
import random as rn
import matplotlib.pyplot as plt
# Load files
from params import *

def note_stats_master(df, df_folder):
    #for spp in species_list:
        #temp_df = df[df['Species'] == 'F. ' + spp]
    temp_df = df[numerical_columns_with_locname]

    max_loc_df = temp_df.drop(columns = ['Note']).groupby(['Species', 'Location']).max()
    max_loc_df.columns = 'Max ' + max_loc_df.columns
    max_loc_df = max_loc_df.round(decimals = 3)
    
    min_loc_df = temp_df.drop(columns = ['Note']).groupby(['Species', 'Location']).min()
    min_loc_df.columns = 'Min ' + min_loc_df.columns
    min_loc_df = min_loc_df.round(decimals = 3)

    max_note_df = temp_df.groupby(['Species', 'Location', 'Note']).max()
    max_note_df.columns = 'Max ' + max_note_df.columns
    max_note_df = max_note_df.round(decimals = 3)

    min_note_df = temp_df.groupby(['Species', 'Location', 'Note']).min()
    min_note_df.columns = 'Max ' + min_note_df.columns
    min_note_df = min_note_df.round(decimals = 3)

    max_species_df = temp_df.drop(columns = ['Note', 'Location']).groupby(['Species']).max()
    max_species_df.columns = 'Max ' + max_species_df.columns
    max_species_df = max_species_df.round(decimals = 3)

    min_species_df = temp_df.drop(columns = ['Note', 'Location']).groupby(['Species']).min()
    min_species_df.columns = 'Max ' + min_species_df.columns
    min_species_df = min_species_df.round(decimals = 3)
    
    max_loc_df.to_csv(os.path.join(df_folder, 'max_loc_df_master.csv'), index = True)
    min_loc_df.to_csv(os.path.join(df_folder, 'min_loc_df_master.csv'), index = True)
    max_note_df.to_csv(os.path.join(df_folder, 'max_note_df_master.csv'), index = True)
    min_note_df.to_csv(os.path.join(df_folder, 'min_note_df_master.csv'), index = True)
    max_species_df.to_csv(os.path.join(df_folder, 'max_species_df_master.csv'), index = True)
    min_species_df.to_csv(os.path.join(df_folder, 'min_species_df_master.csv'), index = True)

def note_stats_file(df, df_folder, cols, cols_test):
    #for spp in species_list:
        #temp_df = df[df['Species'] == 'F. ' + spp]
    temp_df = df[cols]
    #temp_df_plot = cols_test
    #temp_df.plot.box(cols_test)
    #cols_test = cols.remove('Species')
    #cols_test = cols_test.remove('Location')
    max_loc_df = temp_df.groupby(['Species', 'Location']).max()
    max_loc_df.columns = 'Max ' + max_loc_df.columns
    max_loc_df = max_loc_df.round(decimals = 3)
    
    
    
    min_loc_df = temp_df.groupby(['Species', 'Location']).min()
    min_loc_df.columns = 'Min ' + min_loc_df.columns
    min_loc_df = min_loc_df.round(decimals = 3)

    max_species_df = temp_df.drop(columns = ['Location']).groupby(['Species']).max()
    max_species_df.columns = 'Max ' + max_species_df.columns
    max_species_df = max_species_df.round(decimals = 3)

    min_species_df = temp_df.drop(columns = ['Location']).groupby(['Species']).min()
    min_species_df.columns = 'Max ' + min_species_df.columns
    min_species_df = min_species_df.round(decimals = 3)

    #max_note_df = max_note_df.groupby(['Note'])
    #print(max_note_df)
    max_loc_df.to_csv(os.path.join(df_folder, 'max_loc_df_file.csv'), index = True)
    min_loc_df.to_csv(os.path.join(df_folder, 'min_loc_df_file.csv'), index = True)
    max_species_df.to_csv(os.path.join(df_folder, 'max_species_df_file.csv'), index = True)
    min_species_df.to_csv(os.path.join(df_folder, 'min_species_df_file.csv'), index = True)
    