# Loading packages
import os
import numpy as np
import pandas as pd
import random as rn
import seaborn as sn
import matplotlib.pyplot as plt
# Load files
from params import *

def note_stats_master(df, df_folder):
    #for spp in species_list:
        #temp_df = df[df['Species'] == 'F. ' + spp]
    temp_df = df[numerical_columns_with_locname]

    for ii in range(len(numerical_columns)):
        fig = plt.figure(figsize = (7, 7))
        col = numerical_columns[ii]
        sns_plot = sn.violinplot(x = 'Species', y = col, data = temp_df)

        fig_name = 'species_'+col
        sns_plot.set(title = fig_name)
        if not os.path.exists(os.path.join(df_folder, 'stats_plots_notes')):
            os.mkdir(os.path.join(df_folder, 'stats_plots_notes'))
        save_path = os.path.join(df_folder, 'stats_plots_notes', fig_name + '.png')
        plt.savefig(save_path)
        plt.close(fig)
    
    for spp in species_list:
        sp_name = 'F. ' + spp
        temp_df_sp = temp_df[temp_df['Species'] == sp_name]
        #print(temp_df_sp.head())
        for ii in range(len(numerical_columns)):
            fig = plt.figure(figsize = (7, 7))
            col = numerical_columns[ii]
            sns_plot = sn.violinplot(x = 'Location', y = col, data = temp_df_sp)
            plt.xticks(rotation= 45)

            fig_name = 'species_' + spp + '_locations_'+col
            sns_plot.set(title = fig_name)
            if not os.path.exists(os.path.join(df_folder, 'stats_plots_notes', spp)):
                os.mkdir(os.path.join(df_folder, 'stats_plots_notes', spp))
            save_path = os.path.join(df_folder, 'stats_plots_notes', spp, fig_name + '.png')
            plt.savefig(save_path)
            plt.close(fig)

def note_stats_file(df, df_folder, cols, cols_test):
    #for spp in species_list:
        #temp_df = df[df['Species'] == 'F. ' + spp]
    temp_df = df[cols]
    
    for ii in range(len(cols_test)):
        fig = plt.figure(figsize = (7, 7))
        col = cols_test[ii]
        sns_plot = sn.violinplot(x = 'Species', y = col, data = temp_df)

        fig_name = 'species_'+col
        sns_plot.set(title = fig_name)
        if not os.path.exists(os.path.join(df_folder, 'stats_plots_files')):
            os.mkdir(os.path.join(df_folder, 'stats_plots_files'))
        save_path = os.path.join(df_folder, 'stats_plots_files', fig_name + '.png')
        plt.savefig(save_path)
        plt.close(fig)
    
    for spp in species_list:
        sp_name = 'F. ' + spp
        temp_df_sp = temp_df[temp_df['Species'] == sp_name]
        #print(temp_df_sp.head())
        for ii in range(len(cols_test)):
            fig = plt.figure(figsize = (7, 7))
            col = cols_test[ii]
            sns_plot = sn.violinplot(x = 'Location', y = col, data = temp_df_sp)
            plt.xticks(rotation= 45)

            fig_name = 'locations_'+col
            sns_plot.set(title = fig_name)
            if not os.path.exists(os.path.join(df_folder, 'stats_plots_files', spp)):
                os.mkdir(os.path.join(df_folder, 'stats_plots_files', spp))
            save_path = os.path.join(df_folder, 'stats_plots_files', spp, fig_name + '.png')
            plt.savefig(save_path)
            plt.close(fig)
    