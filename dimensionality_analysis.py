import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import umap.plot
import matplotlib.pyplot as plt
#import umap.umap_ as umap

from params import *
from plots import scatter_plot

def pca_analysis(fdf, X, y, title, filename):
    pca = PCA(n_components = 2, random_state = 42)
    pca_dataset = X.apply(pd.to_numeric)
    X_pca = pca.fit_transform(pca_dataset)
    #print(X_pca.shape)
    #print(X.index)
    principalDf = pd.DataFrame(data = X_pca,
            columns = ['Principal Component 1', 'Principal Component 2'])
    final_df = pd.concat([principalDf, y], axis = 1)
    print(final_df.shape)
    target_names = np.unique(y)
    #loading_scores = pd.Series(pca.components_[0], index = X.index)
    #print(loading_scores)
    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    bibdx = np.where((final_df['Species'] == 'F. pennanti') & (final_df['Principal Component 1'] < 0))
    blurb = fdf.loc[bibdx]
    print(blurb['Location'])

    ax = scatter_plot(final_df, target_names)
    ax.set_xlabel('PC 1: ' + str(round(pca.explained_variance_ratio_[0], 4) * 100) + '%')
    ax.set_ylabel('PC 2: ' + str(round(pca.explained_variance_ratio_[1], 4) * 100) + '%')
    ax.set_title('\n' + title + '\n')
    save_path = os.path.join(PROJECT_PATH, 'Figures/' + filename)
    plt.savefig(save_path)
    #plt.close(fig)

def umap_analysis(X, y, title, filename):
    n = 20
    d = 0.5
    m = 'euclidean'
    title = ('UMAP with n_neighbors = {}, min_dist = {}, metric = {}'.format(n, d, m))
    embedding=umap.UMAP(n_neighbors= n, n_components = 2, min_dist = d, metric = m, random_state = 42).fit(X)
    umap.plot.points(embedding, labels = y, theme = 'viridis')
    save_path = os.path.join(PROJECT_PATH, 'Figures/' + filename)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('\n' + title + '\n')
    plt.savefig(save_path)
    #plt.close(fig)

def lda_analysis(fdf, X, y, title, filename):
    lda = LinearDiscriminantAnalysis(n_components = 2)
    lda_dataset = X.apply(pd.to_numeric)
    X_lda = lda.fit_transform(lda_dataset, y)
    principalDf = pd.DataFrame(data = X_lda,
            columns = ['Principal Component 1', 'Principal Component 2'])
    final_df = pd.concat([principalDf, y], axis = 1)
    target_names = np.unique(y)

    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
          % str(lda.explained_variance_ratio_))

    bibdx = np.where((final_df['Species'] == 'F. pennanti') & (final_df['Principal Component 1'] < 0))
    blurb = fdf.loc[bibdx]
    print(blurb['Location'])

    ax = scatter_plot(final_df, target_names)
    ax.set_xlabel('PC 1: ' + str(round(lda.explained_variance_ratio_[0], 4) * 100) + '%')
    ax.set_ylabel('PC 2: ' + str(round(lda.explained_variance_ratio_[1], 4) * 100) + '%')
    ax.set_title('\n' + title + '\n')
    save_path = os.path.join(PROJECT_PATH, 'Figures/' + filename)
    plt.savefig(save_path)
    #plt.close(fig)
