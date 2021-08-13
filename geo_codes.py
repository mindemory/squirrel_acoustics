import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import DistanceMetric

from params import *
from plots import linear_reg
#selected_features = ['Median PFC Max Freq (Hz)', 'Median PFC Min Freq (Hz)', 'Median BW 90% (Hz)', 'Median Agg Entropy (bits)']

def geo_distance(selected_features, data, type):
    data_spp_list = data['Species'].unique()

    for spp in data_spp_list:
        temp_df = data[data['Species'] == spp].reset_index(drop = True)
        temp_df['Latitude_radians'] = np.radians(temp_df['Latitude'])
        temp_df['Longitude_radians'] = np.radians(temp_df['Longitude'])

        geodist = DistanceMetric.get_metric('haversine')
        geo_distance_matrix = geodist.pairwise(temp_df[['Latitude_radians','Longitude_radians']].to_numpy())*6373
        geo_distance_df = pd.DataFrame(geo_distance_matrix,  columns=temp_df.File_name.unique(), index=temp_df.File_name.unique())
        geo_distance_df = geo_distance_df.stack().reset_index()
        geo_distance_df.columns = ['Loc1', 'Loc2', 'Haversine_distance']
        geo_distance_df['ordered_cols'] = geo_distance_df.apply(lambda x: '-'.join(sorted([x['Loc1'],x['Loc2']])),axis=1)
        geo_distance_df = geo_distance_df.drop_duplicates(['ordered_cols'])
        geo_distance_df.drop(['ordered_cols'], axis=1, inplace=True)#.reset_index(drop = True)
        spp_distance_df = pd.DataFrame()

        for feat in selected_features:
            #feat = 'Median ' + f
            metricdist = DistanceMetric.get_metric('euclidean')
            metric_distance_matrix = metricdist.pairwise(temp_df[[feat]].to_numpy())
            metric_distance_df = pd.DataFrame(metric_distance_matrix,  columns=temp_df.File_name.unique(), index=temp_df.File_name.unique())
            metric_distance_df = metric_distance_df.stack().reset_index()
            metric_distance_df.columns = ['Loc1', 'Loc2', feat + '_Acoustic_distance']
            metric_distance_df['ordered_cols'] = metric_distance_df.apply(lambda x: '-'.join(sorted([x['Loc1'],x['Loc2']])),axis=1)
            metric_distance_df = metric_distance_df.drop_duplicates(['ordered_cols'])
            metric_distance_df.drop(['ordered_cols'], axis=1, inplace=True)#.reset_index(drop = True)

            spp_distance_df = pd.merge(geo_distance_df, metric_distance_df)
            linear_reg(spp_distance_df['Haversine_distance'], spp_distance_df[feat + '_Acoustic_distance'], feat, spp, type)
        #col = spp_distance_df.drop[['Loc1']]
        #spp_distance_df['mean_metric'] = spp_distance_df
        #spp_distance_df.to_csv(os.path.join(PROJECT_PATH, 'dataframes/' + spp + '_distance_df.csv'))

        #linear_reg(spp_distance_df['Haversine_distance'], spp_distance_df['Acoustic_distance'], feat, spp)
