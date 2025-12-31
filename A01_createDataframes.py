
# Loading packages
import os
import shutil
import numpy as np
import pandas as pd

# Loading files and functions
import pre_proc_func as ppf
from params import *

def main():
    print("--- Regenerating all Dataframes (A01) ---")
    
    # Define Dataframe Folder
    df_folder = os.path.join(PROJECT_PATH, 'dataframes')
    if not os.path.exists(df_folder):
        os.mkdir(df_folder)
    else:
        # User requested to delete existing dataframes
        print("Cleaning existing dataframes...")
        files_to_remove = ['master_df.csv', 'master_good_df.csv', 'summary_df.csv',
                           'master_random_df.csv', 'file_df.csv', 'file_good_df.csv', 'file_random_df.csv']
        for f in files_to_remove:
            path = os.path.join(df_folder, f)
            if os.path.exists(path):
                os.remove(path)
                print(f"Deleted {f}")

    # Load Location File (Logic from dataframes_creator.py)
    location_path = os.path.join(PROJECT_PATH, 'locations_dec.csv')
    if not os.path.exists(location_path):
        print(f"Error: Location file not found at {location_path}")
        return

    location_df = pd.read_csv(location_path)
    # Add extension to file names wherever needed
    for i in range(location_df.shape[0]):
        fname = location_df.loc[i, '12_Audio_file_name']
        if isinstance(fname, str):
                location_df.loc[i, '12_Audio_file_name'] = fname + '.wav'
    


    # 1. Create master_df and master_good_df
    print('Creating master_df...')
    master_df = ppf.generate_master_df(species_list, location_df)
    master_good_df = master_df[master_df['Quality'] == 'G']

    master_df.to_csv(os.path.join(df_folder, 'master_df.csv'), index = False)
    master_good_df.to_csv(os.path.join(df_folder, 'master_good_df.csv'), index = False)

    # 2. Create summary_df
    print('Creating summary_df...')
    df_by_species = master_df.groupby('Species')
    df_by_species_good = master_df[master_df['Quality'] == 'G'].groupby('Species')
    summary_df_1 = df_by_species.agg({'File_name': 'nunique', 'Note': ['count', 'nunique']})
    summary_df_1.columns = ['_'.join(col) for col in summary_df_1.columns]
    summary_df_2 = df_by_species_good.agg({'Note': ['count','nunique']})
    summary_df_2.columns = ['_good_'.join(col) for col in summary_df_2.columns]
    summary_df = summary_df_1.join(summary_df_2)
    summary_df.to_csv(os.path.join(df_folder, 'summary_df.csv'))

    # 3. Create master_random_df
    print('Creating randomly sampled dataframe (master_random_df)...')
    master_random_df = pd.DataFrame()
    beg_file = master_good_df['File_name'].unique()
    
    for bf in beg_file:
        sp_df = master_good_df[master_good_df['File_name'] == bf]
        
        # Get threshold for this species
        if not sp_df.empty:
            spp = sp_df['Species'].iloc[0]
            threshold_value = sample_threshold_dict.get(spp, 30) # Default if lookup fails
            
            if sp_df.shape[0] > threshold_value:
                sp_df = sp_df.sample(threshold_value, random_state = 42)
            master_random_df = pd.concat([master_random_df, sp_df], ignore_index = True)
            
    master_random_df.to_csv(os.path.join(df_folder, 'master_random_df.csv'), index = False)

    # 4. Create File Dataframes
    print('Creating file dataframes...')
    file_df = ppf.generate_file_df(master_df, location_df)
    file_good_df = ppf.generate_file_df(master_good_df, location_df)
    file_random_df = ppf.generate_file_df(master_random_df, location_df)

    file_df.to_csv(os.path.join(df_folder, 'file_df.csv'), index = False)
    file_good_df.to_csv(os.path.join(df_folder, 'file_good_df.csv'), index = False)
    file_random_df.to_csv(os.path.join(df_folder, 'file_random_df.csv'), index = False)
    
    print("\nSuccess! All dataframes regenerated in:", df_folder)

if __name__ == "__main__":
    main()
