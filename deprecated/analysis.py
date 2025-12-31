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

df_path = os.path.join(PROJECT_PATH, 'dataframes')

# Load dataframes
print("Loading dataframes")
master_df = pd.read_csv(os.path.join(df_path, 'master_df.csv'))
master_good_df = pd.read_csv(os.path.join(df_path, 'master_good_df.csv'))
master_random_df = pd.read_csv(os.path.join(df_path, 'master_random_df.csv'))
file_df = pd.read_csv(os.path.join(df_path, 'file_df.csv'))
file_good_df = pd.read_csv(os.path.join(df_path, 'file_good_df.csv'))
file_random_df = pd.read_csv(os.path.join(df_path, 'file_random_df.csv'))
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
#geo_distance(file_df.drop(columns = cat_columns).columns, file_df, type = 'notes')
geo_distance(file_df.drop(columns = cat_columns).columns, file_df, type = 'files')

master_df_bout_version = pd.DataFrame()
for bf in master_df['File_name'].unique():
    temp_df = master_df[master_df['File_name'] == bf].reset_index(drop = True)
    temp_df = temp_df.drop([0]).reset_index(drop = True)
    master_df_bout_version = pd.concat([master_df_bout_version, temp_df])

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


print("Saving correlation plot")
corr_note_df = master_good_df[acoustic_features]
corr_note_Matrix = corr_note_df.corr()
heatmap(corr_note_Matrix, suptitle = 'Heat Map for correlations', filename = 'master_correl_heat_map_notes.png')

corr_file_df = file_good_df.drop(columns = cat_columns)
corr_file_Matrix = corr_file_df.corr()
heatmap(corr_file_Matrix, suptitle = 'Heat Map for correlations', filename = 'master_correl_heat_map_files.png')
print()

# print('NOTE LEVEL ANALYSIS')
# print('Running Classifiers')
# X = master_random_df[acoustic_features]
# sc = StandardScaler()
# scaled_features = sc.fit_transform(X)
# X_scaled = pd.DataFrame(scaled_features, index = X.index, columns = X.columns)
# y = master_random_df['Species']

# print('Logistic Regression')
# logreg(X_scaled, y, title = 'Feature Importance Logistic Regression (notes)',
#     filename = 'logistic_regression_feature_importance_notes.png')
# print()

# print('Decision Tree Classifier')
# dtc(X_scaled, y, title = 'Feature Importance Decision Tree Classifier (notes)',
#     filename = 'decision_tree_classifier_feature_importance_notes.png')
# print()

# print('Random Forest Classifier')
# rtf(X_scaled, y, title = 'Feature Importance Random Forest Classifier (notes)',
#     filename = 'random_forest_classifier_feature_importance_notes.png')
# print()

# print('Running PCA, UMAP, and LDA')
# #pca_analysis(X_scaled[selected_features_notes], y, title = 'PCA for species (notes)', filename = 'pca_notes.png')
# #umap_analysis(X_scaled[selected_features_notes], y, title = 'UMAP for species (notes)', filename = 'umap_notes.png')
# #lda_analysis(X_scaled[selected_features_notes], y, title = 'LDA for species (notes)', filename = 'lda_notes.png')
# print()

print('FILE LEVEL ANALYSIS')
print('Running Classifiers')
X = file_df.drop(columns = cat_columns)
#file_df_fname = file_df[columns = ['File_name']]
sc = StandardScaler()
scaled_features = sc.fit_transform(X)
X_scaled = pd.DataFrame(scaled_features, index = X.index, columns = X.columns)
y = file_df['Species']
#print(X.head())
print('Logistic Regression')
logreg(X_scaled, y, title = 'Feature Importance Logistic Regression (files)',
    filename = 'logistic_regression_feature_importance_files.png')
print()

print('Decision Tree Classifier')
dtc(X_scaled, y, title = 'Feature Importance Decision Tree Classifier (files)',
    filename = 'decision_tree_classifier_feature_importance_files.png')
print()

print('Random Forest Classifier')
rtf(X_scaled, y, title = 'Feature Importance Random Forest Classifier (files)',
    filename = 'random_forest_classifier_feature_importance_files.png')
print()

print('Running PCA, UMAP, and LDA')
pca_analysis(file_df, X_scaled, y, title = 'PCA for species (files)', filename = 'pca_files_all_feats.png')
umap_analysis(file_df, X_scaled, y, title = 'UMAP for species (files)', filename = 'umap_files_all_feats.png')
lda_analysis(file_df, X_scaled, y, title = 'LDA for species (files)', filename = 'lda_files_all_feats.png')

pca_analysis(file_df, X_scaled[selected_features_files], y, title = 'PCA for species (files)', filename = 'pca_files_3_feats.png')
umap_analysis(file_df, X_scaled[selected_features_files], y, title = 'UMAP for species (files)', filename = 'umap_files_3_feats.png')
lda_analysis(file_df, X_scaled[selected_features_files], y, title = 'LDA for species (files)', filename = 'lda_files_3_feats.png')

print()
