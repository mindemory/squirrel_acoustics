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
    
    principalDf = pd.DataFrame(data = X_pca,
            columns = ['Principal Component 1', 'Principal Component 2'])
    final_df = pd.concat([principalDf, y], axis = 1)
    print(final_df.shape)
    target_names = np.unique(y)
    
    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    bibdx = np.where((final_df['Species'] == 'F. pennanti') & (final_df['Principal Component 1'] < 0))
    blurb = fdf.loc[bibdx]
    #print(blurb[['Location', 'File_name']])

    ax = scatter_plot(final_df, target_names)
    ax.set_xlabel('PC 1: ' + str(round(pca.explained_variance_ratio_[0] * 100, 2)) + '%')
    ax.set_ylabel('PC 2: ' + str(round(pca.explained_variance_ratio_[1] * 100, 2)) + '%')
    ax.set_title('\n' + title + '\n')
    if not os.path.exists(os.path.join(PROJECT_PATH, 'Figures', 'dimred')):
            os.mkdir(os.path.join(PROJECT_PATH, 'Figures', 'dimred'))
    if 'notes' in filename:
        save_path = os.path.join(PROJECT_PATH, 'Figures', 'dimred', 'notes', filename)
        if not os.path.exists(os.path.join(PROJECT_PATH, 'Figures', 'dimred', 'notes')):
            os.mkdir(os.path.join(PROJECT_PATH, 'Figures', 'dimred', 'notes'))
    elif 'files' in filename:
        save_path = os.path.join(PROJECT_PATH, 'Figures', 'dimred', 'files', filename)
        if not os.path.exists(os.path.join(PROJECT_PATH, 'Figures', 'dimred', 'files')):
            os.mkdir(os.path.join(PROJECT_PATH, 'Figures', 'dimred', 'files'))
    else:
        save_path = os.path.join(PROJECT_PATH, 'Figures', 'dimred', filename)
        
    plt.savefig(save_path)
    

def umap_analysis(fdf, X, y, title, filename):
    n = 100 #20
    d = 1 #0.5
    m = 'euclidean'
    title = ('UMAP with n_neighbors = {}, min_dist = {}, metric = {}'.format(n, d, m))
    embedding=umap.UMAP(n_neighbors= n, n_components = 2, min_dist = d, metric = m, random_state = 42).fit(X)
    umap.plot.points(embedding, labels = y, theme = 'viridis')
    if not os.path.exists(os.path.join(PROJECT_PATH, 'Figures', 'dimred')):
            os.mkdir(os.path.join(PROJECT_PATH, 'Figures', 'dimred'))
    if 'notes' in filename:
        save_path = os.path.join(PROJECT_PATH, 'Figures', 'dimred', 'notes', filename)
        if not os.path.exists(os.path.join(PROJECT_PATH, 'Figures', 'dimred', 'notes')):
            os.mkdir(os.path.join(PROJECT_PATH, 'Figures', 'dimred', 'notes'))
    elif 'files' in filename:
        save_path = os.path.join(PROJECT_PATH, 'Figures', 'dimred', 'files', filename)
        if not os.path.exists(os.path.join(PROJECT_PATH, 'Figures', 'dimred', 'files')):
            os.mkdir(os.path.join(PROJECT_PATH, 'Figures', 'dimred', 'files'))
    else:
        save_path = os.path.join(PROJECT_PATH, 'Figures', 'dimred', filename)
        
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('\n' + title + '\n')
    plt.savefig(save_path)

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
    #print(blurb[['Location', 'File_name']])

    ax = scatter_plot(final_df, target_names)
    ax.set_xlabel('LD 1: ' + str(round(lda.explained_variance_ratio_[0] * 100, 2)) + '%')
    ax.set_ylabel('LD 2: ' + str(round(lda.explained_variance_ratio_[1] * 100, 2)) + '%')
    ax.set_title('\n' + title + '\n')
    if not os.path.exists(os.path.join(PROJECT_PATH, 'Figures', 'dimred')):
            os.mkdir(os.path.join(PROJECT_PATH, 'Figures', 'dimred'))
    if 'notes' in filename:
        save_path = os.path.join(PROJECT_PATH, 'Figures', 'dimred', 'notes', filename)
        if not os.path.exists(os.path.join(PROJECT_PATH, 'Figures', 'dimred', 'notes')):
            os.mkdir(os.path.join(PROJECT_PATH, 'Figures', 'dimred', 'notes'))
    elif 'files' in filename:
        save_path = os.path.join(PROJECT_PATH, 'Figures', 'dimred', 'files', filename)
        if not os.path.exists(os.path.join(PROJECT_PATH, 'Figures', 'dimred', 'files')):
            os.mkdir(os.path.join(PROJECT_PATH, 'Figures', 'dimred', 'files'))
    else:
        save_path = os.path.join(PROJECT_PATH, 'Figures', 'dimred', filename)
        
    plt.savefig(save_path)
