
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance
import umap

from params import PROJECT_PATH, acoustic_features

# --- Configuration ---
TARGET_SPECIES = ['F. palmarum', 'F. tristriatus', 'F. pennanti']
MIN_FILES_PER_LOCATION = 10
SEED = 42
TOP_N_FEATURES = 5

# Color Palette (Consistent with F03)
SPECIES_COLORS = {
    'F. palmarum': '#D55E00',      # Vermillion
    'F. pennanti': '#E69F00',      # Orange
    'F. tristriatus': '#009E73',   # Bluish Green
    'F. sublineatus': '#CC79A7'    # Reddish Purple
}

# --- Helper Functions ---

def clear_output_dir(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(directory)

def get_feature_importance(X, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, class_weight='balanced')
    rf.fit(X, y)
    
    # Use permutation importance for robustness
    result = permutation_importance(rf, X, y, n_repeats=10, random_state=SEED, n_jobs=-1)
    
    importances = result.importances_mean
    feature_names = X.columns
    
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    return feature_importance_df

def plot_dimred(X_transformed, y, title, save_path, explained_variance=None, loadings=None, feature_names=None, fixed_scale=True):
    # Determine layout based on whether we need to show loadings
    if loadings is not None and feature_names is not None:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        ax_scatter = axes[0]
        ax_loadings = axes[1]
    else:
        fig, ax_scatter = plt.subplots(1, 1, figsize=(10, 8))
        ax_loadings = None
    
    # --- Scatter Plot ---
    unique_labels = sorted(y.unique())
    n_colors = len(unique_labels)
    palette = "tab10" if n_colors <= 10 else "tab20"
        
    sns.scatterplot(x=X_transformed[:, 0], y=X_transformed[:, 1], hue=y, palette=palette, 
                    s=15, alpha=0.7, edgecolor='none', marker='o', ax=ax_scatter)
    
    ax_scatter.set_title(title)
    
    # Set Axis Limits as requested (only if fixed_scale is True)
    if fixed_scale:
        ax_scatter.set_xlim(-5, 5)
        ax_scatter.set_ylim(-5, 5)
    
    # Labels with Variance
    xlabel = "Component 1"
    ylabel = "Component 2"
    if explained_variance is not None:
        if len(explained_variance) >= 1:
            xlabel += f" ({explained_variance[0]*100:.1f}%)"
        if len(explained_variance) >= 2:
            ylabel += f" ({explained_variance[1]*100:.1f}%)"
    
    ax_scatter.set_xlabel(xlabel)
    ax_scatter.set_ylabel(ylabel)
    ax_scatter.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Location")
    ax_scatter.grid(True, linestyle='--', alpha=0.3)

    # --- Loadings Plot (if applicable) ---
    if ax_loadings is not None:
        n_comps = loadings.shape[1]
        
        # Create DataFrame for easy plotting
        # We only care about first 2 components for 2D plot context usually
        cols = [f'Comp {i+1}' for i in range(min(n_comps, 2))]
        loadings_df = pd.DataFrame(loadings[:, :2], index=feature_names, columns=cols)
        
        # Plot coefficients
        loadings_df.plot(kind='barh', ax=ax_loadings, colormap='coolwarm', width=0.8)
        
        ax_loadings.set_title("Feature Contributions (Loadings)")
        ax_loadings.axvline(0, color='black', linewidth=0.8)
        ax_loadings.set_xlabel("Coefficient Value")
        ax_loadings.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_boxplots(df, features, target_col, save_dir, species_name):
    num_features = len(features)
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()
    
    unique_labels = df[target_col].unique()
    n_colors = len(unique_labels)
    palette = "tab10" if n_colors <= 10 else "tab20"
    
    for i, feature in enumerate(features):
        ax = axes[i]
        sns.boxplot(x=target_col, y=feature, data=df, showfliers=False, palette=palette, ax=ax)
        
        means = df.groupby(target_col)[feature].mean()
        locations = [label.get_text() for label in ax.get_xticklabels()]
        mean_values = [means[loc] for loc in locations]
        ax.plot(range(len(locations)), mean_values, marker='D', color='white', markeredgecolor='black', linestyle='none', markersize=7, label='Mean')

        ax.set_title(feature)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        
        if feature == 'Inter_note_difference (s)':
             ax.set_ylim(0, 2)
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle(f"{species_name}: Top {num_features} Features by Location")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'differentiating_features_boxplots.png'), dpi=300)
    plt.close()

# --- Main Execution ---

def main():
    print("Loading Data (Master Good DF)...")
    df_folder = os.path.join(PROJECT_PATH, 'dataframes')
    master_df = pd.read_csv(os.path.join(df_folder, 'master_good_df.csv'))
    
    if 'Inter_note_difference (s)' in master_df.columns:
        print("Filtering negative inter-note differences...")
        before_count = len(master_df)
        master_df = master_df[master_df['Inter_note_difference (s)'] >= 0]
        print(f"Removed {before_count - len(master_df)} rows with negative intervals.")

    # Subsample notes per file (Max 60)
    print("Subsampling max 60 notes per file...")
    before_count = len(master_df)
    # Using group_keys=False to keep index clean, sample with SEED
    master_df = master_df.groupby('File_name', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 60), random_state=SEED))
    print(f"Dataset size: {before_count} -> {len(master_df)}")

    base_save_dir = os.path.join(PROJECT_PATH, 'Figures', 'geo_variation')
    
    summary_results = []

    for species_name in TARGET_SPECIES:
        print(f"\n--- Analyzing {species_name} ---")
        
        species_dir = os.path.join(base_save_dir, species_name.replace(' ', '_').replace('.', ''))
        clear_output_dir(species_dir)
        
        spp_df = master_df[master_df['Species'] == species_name].copy()
        
        if spp_df.empty:
            continue
            
        loc_file_counts = spp_df.groupby('Location')['File_name'].nunique()
        valid_locations = loc_file_counts[loc_file_counts >= MIN_FILES_PER_LOCATION].index.tolist()
        
        print(f"Valid Locations ({len(valid_locations)}): {valid_locations}")
        
        if len(valid_locations) < 2:
            print("Not enough locations. Skipping.")
            continue
            
        spp_df = spp_df[spp_df['Location'].isin(valid_locations)]
        X = spp_df[acoustic_features]
        y = spp_df['Location']
        
        if X.isna().any().any():
             valid_indices = X.dropna().index
             X = X.loc[valid_indices]
             y = y.loc[valid_indices]
             spp_df = spp_df.loc[valid_indices]

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        print("Evaluating Classifier...")
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=SEED, stratify=y)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, class_weight='balanced')
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Test Accuracy: {acc:.4f}")
        with open(os.path.join(species_dir, 'classification_report.txt'), 'w') as f:
            f.write(f"Species: {species_name}\nN={len(spp_df)}\nAccuracy: {acc:.4f}\n\n{report}")
        
        summary_results.append({
            'Species': species_name,
            'N': len(spp_df),
            'Locations': len(valid_locations),
            'Accuracy': acc
        })

        print("Selecting Top 5 Features (Training on Full Data)...")
        importance_df = get_feature_importance(X_scaled, y)
        importance_df.to_csv(os.path.join(species_dir, 'feature_ranking.csv'), index=False)
        
        top_5_features = importance_df['Feature'].head(TOP_N_FEATURES).tolist()
        print(f"Top 5 Features: {top_5_features}")
        
        X_top5 = X_scaled[top_5_features]

        print("Generating Consolidated Boxplots...")
        plot_boxplots(spp_df, top_5_features, 'Location', species_dir, species_name)
        
        print("Generating PCA/UMAP/LDA for Top 5...")
        
        # PCA
        print("Running PCA (All Features)...")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        # Transpose components for shape (n_features, n_components)
        pca_loadings = pca.components_.T 
        plot_dimred(X_pca, y, f"{species_name}: PCA - All Features", 
                    os.path.join(species_dir, 'PCA_AllFeatures.png'), 
                    explained_variance=pca.explained_variance_ratio_,
                    loadings=pca_loadings,
                    feature_names=X.columns,
                    fixed_scale=True)

        # LDA
        n_classes = len(valid_locations)
        n_lda_components = min(len(X.columns), n_classes - 1)
        if n_lda_components > 0:
            print("Running LDA (All Features)...")
            lda = LinearDiscriminantAnalysis(n_components=n_lda_components)
            X_lda = lda.fit_transform(X_scaled, y)
            
            # Sklearn LDA scalings_ is already (n_features, n_components)
            lda_loadings = lda.scalings_
            
            if n_lda_components == 1:
                X_plot = np.zeros((X_lda.shape[0], 2))
                X_plot[:, 0] = X_lda[:, 0]
                plot_title = "LDA (1D) - All Features"
            else:
                X_plot = X_lda[:, :2]
                plot_title = "LDA (2D) - All Features"
            
            plot_dimred(X_plot, y, f"{species_name}: {plot_title}", 
                        os.path.join(species_dir, 'LDA_AllFeatures.png'), 
                        explained_variance=lda.explained_variance_ratio_,
                        loadings=lda_loadings,
                        feature_names=X.columns,
                        fixed_scale=True)

        # UMAP
        print("Running UMAP (Top 5)...")
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=SEED)
        X_umap = reducer.fit_transform(X_top5)
        # UMAP doesn't have simple linear loadings
        plot_dimred(X_umap, y, f"{species_name}: UMAP - Top 5 Features", 
                    os.path.join(species_dir, 'Top5_UMAP.png'),
                    fixed_scale=False)

    if summary_results:
        pd.DataFrame(summary_results).to_csv(os.path.join(base_save_dir, 'geo_classification_summary.csv'), index=False)

if __name__ == "__main__":
    main()
