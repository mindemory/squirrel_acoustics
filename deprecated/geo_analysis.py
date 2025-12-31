import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from params import *
from plots import note_distribution, inter_note_distribution, heatmap
from dimensionality_analysis import pca_analysis, umap_analysis, lda_analysis
from classifiers import logreg, dtc, rtf
from geo_codes import geo_distance
import matplotlib.pyplot as plt
df_path = os.path.join(PROJECT_PATH, 'dataframes')

# Load dataframes
print("Loading dataframes")
master_df = pd.read_csv(os.path.join(df_path, 'master_df.csv'))
#master_good_df = pd.read_csv(os.path.join(df_path, 'master_good_df.csv'))
#master_random_df = pd.read_csv(os.path.join(df_path, 'master_random_df.csv'))
file_df = pd.read_csv(os.path.join(df_path, 'file_df.csv'))
#file_good_df = pd.read_csv(os.path.join(df_path, 'file_good_df.csv'))
#file_random_df = pd.read_csv(os.path.join(df_path, 'file_random_df.csv'))
print()
selected_features_notes = ['PFC Max Freq (Hz)', 'PFC Min Freq (Hz)', 'BW 90% (Hz)']
#selected_features_files = ['Median Delta Freq (Hz)', 'Median High Freq (Hz)', 'Stdev High Freq (Hz)',
#    ]
selected_features_files = ['Median Low Freq (Hz)', 'Median High Freq (Hz)',
       'Median Delta Freq (Hz)', 'Median Delta Time (s)',
       'Median Inter_note_difference (s)', 'Note density (notes per s)',
       'Sub-bout density (sub-bouts per s)', 'Bout density (bouts per s)',
       'Unique note count']
print('Running Geographical Analysis')
#print(file_df.drop(columns = cat_columns).columns)
#geo_distance(file_df.drop(columns = cat_columns).columns, file_df, type = 'notes')
geo_distance(loc_columns, file_df, type = 'files')
