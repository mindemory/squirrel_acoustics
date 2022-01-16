# Loading packages
import os
import numpy as np
import pandas as pd
import random as rn

# Load files
from plots import line_with_error
from params import *


def screen_tables(species_list):
    # The function screens the annotation tables for possible errors like missing columns,
    # empty columns or missing entries and incorrect entries, that need to be fixed manually.
    error = 0
    for spp in species_list:
        print(spp)
        directory_path = os.path.join(PROJECT_PATH, spp)
        annotation_files = os.listdir(directory_path) # list of all annotation files for a given species
        for annotation_file in annotation_files:
            annotation_path = os.path.join(directory_path, annotation_file)
            ann_df = pd.read_csv(annotation_path, delimiter = '\t')

            # Check if any column is absent
            present_columns = ann_df.columns
            for ac_col in accepted_columns:
                if ac_col not in present_columns:
                    error += 1
                    print('Column Missing error')
                    print(annotation_file)
                    print(ac_col)
                    print()

            # Check if a column is empty (Use column name in box braces)
            if ann_df['Note'].isnull().values.any():
                error += 1
                print('Empty Note column error')
                print(annotation_file)
                print(ann_df['Note'].isnull().sum())
                print()

            if ann_df['Quality'].isnull().values.any():
                error += 1
                print('Empty Quality column error')
                print(annotation_file)
                print(ann_df['Quality'].isnull().sum())
                print()

            # Check if Quality column contains desired unique elements
            unique_elements_quality = ann_df['Quality'].unique()
            accepted_elements_quality = ['G', 'P']
            for element in unique_elements_quality:
                if element not in accepted_elements_quality:
                    error += 1
                    print('Quality column error')
                    print(annotation_file)
                    print(element)
                    print()

            # Check if Note column contains desired unique elements
            unique_elements_note = ann_df['Note'].unique()
            for element in unique_elements_note:
                if element not in accepted_elements_note:
                    error += 1
                    print('Note column error')
                    print(annotation_file)
                    print(element)
                    print()

            # Check if Sub-bout column is fine
            sub_bout_array = ann_df['Sub-bout'].to_numpy()
            for i in range(len(sub_bout_array) - 1):
                if sub_bout_array[i] > sub_bout_array[i+1]:
                    error += 1
                    print('Sub-bout column error')
                    print(annotation_file)
                    print(sub_bout_array[i])
                    print()

            # Check if Note is NS given quality = G
            note_series = ann_df['Note'][ann_df['Quality'] == 'G'].to_numpy()
            unaccepted_element = 'NS'
            if unaccepted_element in note_series:
                error += 1
                print('NS in Good quality error')
                print(annotation_file)
                print()

    if error == 0:
        print('If no errors reported, you are good to go!')
        return True
    else:
        return False

def saturation_analysis(metric, species_list, ci = 95):
    rn.seed(42)
    for spp in species_list:
        # load annotation files
        directory_path = os.path.join(PROJECT_PATH, spp)
        save_folder = os.path.join(PROJECT_PATH, 'Figures/Saturation Analysis ' + metric + '/', spp)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        annotation_files = os.listdir(directory_path)
        if spp != 'sublineatus':
            annotation_files = rn.sample(annotation_files, 20) # a random set of 20 files are chosen for saturation analysis

        for annotation_file in annotation_files:
            # load good quality notes from each file
            annotation_path = os.path.join(directory_path, annotation_file)
            ann_df = pd.read_csv(annotation_path, delimiter = '\t')
            df_good = ann_df[ann_df['Quality'] == 'G']

            # Create a dataframe of random notes of different sample sizes
            sample_df = pd.DataFrame()
            row_count = 0
            for i in range(1, df_good.shape[0]):
                sample = df_good.sample(i)
                size_array = i * np.ones(i)
                sample['Sample size'] = size_array
                sample_df = pd.concat([sample_df, sample], ignore_index = True)

                # Compute the lower and upper bound thresholds
                if i == df_good.shape[0] - 1:
                    average = sample[metric].mean()
                    stdev = sample[metric].std()
                    LB = average - z_score * stdev/np.sqrt(df_good.shape[0])
                    UB = average + z_score * stdev/np.sqrt(df_good.shape[0])

            # Compute the threshold sample size based on LB and UB
            thresholding_array = []
            for i in range(1, df_good.shape[0]):
                sample_average = sample_df[sample_df['Sample size'] == i][metric].mean()
                if len(thresholding_array) >= 1:
                    if sample_average < LB or sample_average > UB:
                        thresholding_array = []
                if LB < sample_average < UB:
                    thresholding_array.append(i)
            if len(thresholding_array) > 0:
                print(annotation_file + ': ' + str(thresholding_array[0]))

            # Create plots
            line_with_error(df_good, sample_df, metric, LB, UB, ci, spp, annotation_file, save_folder)

def generate_master_df(species_list, location_df):
    master_df = pd.DataFrame()
    for spp in species_list:
        directory_path = os.path.join(PROJECT_PATH, spp)
        annotation_files = os.listdir(directory_path) # List of all annotation files for a given species

        for annotation_file in annotation_files:
            annotation_path = os.path.join(directory_path, annotation_file)
            ann_df = pd.read_csv(annotation_path, delimiter = '\t')
            ann_df['File_name'] = annotation_file
            ann_df['Species'] = 'F. ' + spp

            # Inter-note distance
            ann_df['Shifted end time'] = ann_df['End Time (s)'].shift(+1).fillna(0)
            ann_df['Inter_note_difference (s)'] = ann_df['Begin Time (s)'] - ann_df['Shifted end time']
            ann_df = ann_df.drop(columns = ['Shifted end time'])

            # Bout and Sub-bout counts
            bout = []
            bout_count = 0
            for difference in ann_df['Inter_note_difference (s)']:
                if difference > bout_difference_dict[spp]:
                    bout_count += 1
                bout.append(bout_count)
            ann_df['Bout'] = bout

            sub_bout = []
            sub_bout_count = 0
            for sub_bout_difference in ann_df['Inter_note_difference (s)']:
                if sub_bout_difference > sub_bout_difference_dict[spp]:
                    sub_bout_count += 1
                sub_bout.append(sub_bout_count)
            ann_df['Sub-bout'] = sub_bout

            # Location, Latitude, Longitude
            for bf in ann_df['Begin File'].unique():
                if bf[:4] == 'Copy':
                    bf = bf[8:]
                ann_df['Location'] = location_df['Site'][location_df['12_Audio_file_name'].str.lower() ==  bf.lower()].to_numpy()[0]
                ann_df['Latitude'] = location_df['lat_3_Location'][location_df['12_Audio_file_name'].str.lower() ==  bf.lower()].to_numpy()[0]
                ann_df['Longitude'] = location_df['long_3_Location'][location_df['12_Audio_file_name'].str.lower() ==  bf.lower()].to_numpy()[0]

            # For NAN rows, make Quality = P
            for column in acoustic_features:
                ann_df.loc[(ann_df[column].isnull()), ['Quality']] = 'P'
                ann_df[column] = ann_df[column].astype('float')

            master_df = master_df.append(ann_df, ignore_index = True)
    return master_df

def generate_file_df(data):

    # Create basic dataframe with all cat columns and median of features
    cat_df = data[cat_columns].drop_duplicates(subset = ['File_name'])

    med_df = data[numerical_columns_with_fname].groupby(['File_name']).median()
    med_df.columns = 'Median ' + med_df.columns

    min_df = data[numerical_columns_with_fname].groupby(['File_name']).min()
    min_df.columns = 'Minimum ' + min_df.columns

    max_df = data[numerical_columns_with_fname].groupby(['File_name']).max()
    max_df.columns = 'Maximum ' + max_df.columns

    avg_df = data[numerical_columns_with_fname].groupby(['File_name']).mean()
    avg_df.columns = 'Mean ' + avg_df.columns

    std_df = data[numerical_columns_with_fname].groupby(['File_name']).std()
    std_df.columns = 'Stdev ' + std_df.columns

    file_df = pd.merge(cat_df, med_df, on='File_name')
    file_df = pd.merge(file_df, min_df, on='File_name')
    file_df = pd.merge(file_df, max_df, on='File_name')
    file_df = pd.merge(file_df, avg_df, on='File_name')
    file_df = pd.merge(file_df, std_df, on='File_name')



    for i in range(file_df.shape[0]):
        ff = file_df.loc[i, 'File_name']
        temp_ff_df = data[data['File_name'] == ff]

        # Compute note, bout, sub-bout density for each file
        annotation_time = temp_ff_df['End Time (s)'].max() - temp_ff_df['Begin Time (s)'].min()
        note_count = temp_ff_df.shape[0]

        if temp_ff_df['Sub-bout'].min() > 0:
            sub_bout_count = temp_ff_df['Sub-bout'].max() - temp_ff_df['Sub-bout'].min()
        else:
            sub_bout_count = temp_ff_df['Sub-bout'].max() - temp_ff_df['Sub-bout'].min() + 1

        if temp_ff_df['Bout'].min() > 0:
            bout_count = temp_ff_df['Bout'].max() - temp_ff_df['Bout'].min()
        else:
            bout_count = temp_ff_df['Bout'].max() - temp_ff_df['Bout'].min() + 1

        file_df.loc[i, 'Note density (notes per s)'] = note_count/annotation_time
        file_df.loc[i, 'Sub-bout density (sub-bouts per s)'] = sub_bout_count/annotation_time
        file_df.loc[i, 'Bout density (bouts per s)'] = bout_count/annotation_time

        # Count of unique note types for each file
        unique_notes_list = temp_ff_df['Note'].unique()
        if 'NS' in unique_notes_list:
            unique_notes_list = np.setdiff1d(unique_notes_list, np.array(['NS']))
        file_df.loc[i, 'Unique note count'] = temp_ff_df['Note'].unique().shape[0]
    #print(file_df.shape)
    #print(file_df.columns)
    #print(file_df.head())
    #print(file_df.describe)
        # Median features by bout and sub-bout
        #temp_ff_bout_df = temp_ff_df.groupby(['Bout']).median()
        #temp_ff_sub_bout_df = temp_ff_df.groupby(['Sub-bout']).median()
        #print(temp_ff_bout_df.columns)
        #bout_num_df = temp_ff_bout_df[[numerical_columns]]
        #sub_bout_num_df = temp_ff_sub_bout_df[[numerical_columns]]
        #print(bout_num_df.head())
    return file_df
