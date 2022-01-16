import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from matplotlib import rc
from params import *

def line_with_error(main_data, sample_data, metric, LB, UB, ci, spp, annotation_file, save_folder):
    sn.set(font_scale = 1)
    fig = plt.figure(figsize = (7, 7))
    sns_plot = sn.lineplot(data = sample_data, x = "Sample size", y = metric, ci = ci)
    plt.plot([0, main_data.shape[0] - 1], [LB, LB], 'k', linewidth = 2)
    plt.plot([0, main_data.shape[0] - 1], [UB, UB], 'k', linewidth = 2)
    sns_plot.set(title = spp + '\n' + annotation_file + '\n Total notes = ' + str(main_data.shape[0]))
    #sns_plot.set(ylim = (0, ))
    save_path = os.path.join(save_folder, annotation_file + '.png')
    plt.savefig(save_path)
    plt.close(fig)

def note_distribution(data, species_list):
    df = pd.DataFrame()
    for spp in species_list:
        Fspp = 'F. ' + spp
        spec_count = pd.DataFrame()
        spec_count[Fspp] = data[data['Species'] == Fspp]['Note'].value_counts()
        df = pd.concat([df, spec_count], axis = 1)
    df = df.fillna(0)
    df = df.astype('int32')
    df['Total'] = df.sum(axis = 1)
    df = df.sort_values(by = 'Total', axis = 0, ascending = False)
    df = df.drop(['Total'], axis = 1)
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 8), dpi = 100)
    df.plot.barh(ax = axes[0], color = ['#98003F', 'orange', '#5D4EA0', '#55ab0f'], stacked = True,
                title = '\nNote abundance and distribution across species\n', rot = 0)
    df.T.plot.bar(ax = axes[1], stacked = True, title = '\nNote distribution within species\n', rot = 45,
                colormap = 'gist_stern')
    axes[1].legend(bbox_to_anchor=(1.25, 0.95))
    save_path = os.path.join(PROJECT_PATH, 'Figures/note_distribution.png')
    plt.savefig(save_path)
    plt.close(fig)

def inter_note_distribution(species_list, master_df_bout_version):
    fig, ax = plt.subplots(nrows = 4, ncols = 1, dpi = 100, figsize = (10, 8), sharex = True)
    for i in range(len(species_list)):
        spp = species_list[i]
        temp_df = master_df_bout_version[master_df_bout_version['Species'] == 'F. ' + spp]['Inter_note_difference (s)']
        temp_df.hist(ax = ax[i], density = 1, bins = bins_dict[spp], xlabelsize = 12, ylabelsize = 11)
        ax[i].set_title('F. ' + spp, fontsize = 14)
        if i == 3:
            ax[i].set_xlabel('Internote differences (s)', fontsize = 12)
        ax[i].set_ylabel('Frequency (%)', fontsize = 12)
        ax[i].set_xlim(-0.1, 2)
        ax[i].set_xticks(np.arange(-0.1, 2, 0.1))
        ax[i].grid(False)
    fig.subplots_adjust(wspace=0, hspace=0.5)
    save_path = os.path.join(PROJECT_PATH, 'Figures/inter_note_distribution.png')
    plt.savefig(save_path)
    plt.close(fig)

def scatter_plot(final_df, target_names):
    sn.set(font_scale=1.4)
    colors = ['#98003F', 'orange', '#55ab0f', '#5D4EA0']
    lw = 2
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(1,1,1)
    for target, color in zip(target_names, colors):
         indicesToKeep = final_df['Species'] == target
         ax.scatter(final_df.loc[indicesToKeep, 'Principal Component 1'],
                    final_df.loc[indicesToKeep, 'Principal Component 2'],
                    c = color, alpha = .6, lw = lw, s = 15)
    ax.legend(target_names)
    return ax

def heatmap(corrMatrix, suptitle, filename):
    ax = plt.figure(figsize = (20, 20))
    mask = np.zeros_like(corrMatrix)
    mask[np.triu_indices_from(mask)] = True
    fig = plt.figure(figsize = (30, 30))
    sn.set(font_scale = 1.6)
    top_corr_features = corrMatrix.index
    sn.heatmap(corrMatrix, annot=True, vmin = -1, vmax = 1, cmap  = 'RdYlGn', linewidths = .5,
               mask = mask, fmt = '.2f')
    plt.suptitle(suptitle, fontsize = 36, y = 0.9)
    save_path = os.path.join(PROJECT_PATH, 'Figures/' + filename)
    plt.savefig(save_path)
    plt.close(ax)

def feat_importance_plot(results, X):
    sn.set(font_scale = 1)
    fig = plt.figure(figsize = (17, 10))
    feat_importances = pd.Series(results.importances_mean, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    return fig

def linear_reg(x, y, feat, spp, type):
    fig = plt.figure()
    plt.scatter(x, y, s = 1)
    plt.xlabel('Geographical distance (kms)')
    plt.ylabel('Acoustic distance (Hz)')
    plt.title(feat + ' ' + spp)
    save_path = os.path.join(PROJECT_PATH, 'Figures/lnreg/' + feat + '_' + spp + '_' + type +'.png')
    plt.savefig(save_path)
    plt.close(fig)
