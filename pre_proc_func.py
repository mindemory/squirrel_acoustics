# Loading packages
import os
import numpy as np
import pandas as pd
import random as rn

# Load files
from plots import line_with_error
from params import *


def screen_tables(PROJECT_PATH, species_list):
    error = 0
    for spp in species_list:
        directory_path = os.path.join(PROJECT_PATH, spp)
        annotation_files = os.listdir(directory_path)
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


def saturation_analysis(PROJECT_PATH, metric, species_list, ci = 95):
    rn.seed(42)
    for spp in species_list:
        # load annotation files
        directory_path = os.path.join(PROJECT_PATH, spp)
        save_folder = os.path.join(PROJECT_PATH, 'Figures/Saturation Analysis ' + metric + '/', spp)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        annotation_files = os.listdir(directory_path)
        if spp != 'sublineatus':
            annotation_files = rn.sample(annotation_files, 20)

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
                size_array = i* np.ones(i)
                sample['Sample size'] = size_array
                sample_df = pd.concat([sample_df, sample])

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

def generate_master_df(PROJECT_PATH, species_list, location_df):
    master_df = pd.DataFrame()
    for spp in species_list:
        directory_path = os.path.join(PROJECT_PATH, spp)
        annotation_files = os.listdir(directory_path)
        #globals()[spp + '_df'] = pd.DataFrame()
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
                if sub_bout_difference > sub_bout_threshold:
                    sub_bout_count += 1
                sub_bout.append(sub_bout_count)
            ann_df['Sub-bout'] = sub_bout

            # Location, Latitude, Longitude
            for bf in ann_df['Begin File'].unique():
                if bf[:4] == 'Copy':
                    bf = bf[8:]
                ann_df['Location'] = location_df['Site'][location_df['12_Audio_file_name'] ==  bf].to_numpy()[0]
                ann_df['Latitude'] = location_df['lat_3_Location'][location_df['12_Audio_file_name'] ==  bf].to_numpy()[0]
                ann_df['Longitude'] = location_df['long_3_Location'][location_df['12_Audio_file_name'] ==  bf].to_numpy()[0]

            # Trim columns
            ann_df = ann_df[master_df_columns]

            # For NAN rows, make Quality = P
            for column in master_df_numerical_columns:
                ann_df.loc[(ann_df[column].isnull()), ['Quality']] = 'P'
                ann_df[column] = ann_df[column].astype('float')

            master_df = master_df.append(ann_df, ignore_index = True)
    return master_df
