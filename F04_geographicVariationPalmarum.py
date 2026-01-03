
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA

from params import PROJECT_PATH, acoustic_features

# --- Configuration ---
SPECIES_NAME = 'F. palmarum'
MIN_FILES_PER_LOCATION = 10
SEED = 42

# User-Requested Location Order
LOCATION_ORDER = ['Colombo', 'Kodaikanal', 'Ottapalam', 'Mettupalayam', 'Bengaluru', 'Vijayawada', 'Tirupati', 'Chennai']

# Define Fixed Colors (tab10)
TAB10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

LOCATION_COLORS = {
    'Colombo': TAB10[0],       # Blue
    'Kodaikanal': TAB10[1],    # Orange
    'Ottapalam': TAB10[2],     # Green
    'Mettupalayam': TAB10[3],  # Red
    'Bengaluru': TAB10[4],     # Purple
    'Vijayawada': TAB10[5],    # Brown
    'Tirupati': TAB10[6],      # Pink
    'Chennai': TAB10[7]        # Gray
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

    plt.close()

def plot_feature_boxplots(df, features, target_col, output_dir, location_order, location_colors):
    # Simplified Boxplots: No significance bars per user request
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    # --- Unit Conversion for Plotting ---
    df_plot = df.copy()
    features_plot = []
    
    for feat in features:
        new_feat = feat
        if '(Hz)' in feat:
            new_feat = feat.replace('(Hz)', '(kHz)')
            df_plot[new_feat] = df_plot[feat] / 1000.0
        else:
            df_plot[new_feat] = df_plot[feat]
        features_plot.append(new_feat)
    
    for i, feature in enumerate(features_plot):
        # Hide fliers to prevent squashing
        sns.boxplot(x=target_col, y=feature, data=df_plot, ax=axes[i], order=location_order, palette=location_colors, showfliers=False)
        means = df_plot.groupby(target_col)[feature].mean()
        
        # Ensure means align with order
        current_means = []
        for loc in location_order:
            if loc in means.index:
                current_means.append(means[loc])
            else:
                current_means.append(np.nan)
                
        axes[i].plot(range(len(location_order)), current_means, 
                     marker='D', color='white', markeredgecolor='black', linestyle='none', markersize=7, label='Mean')

        axes[i].set_title(feature)
        axes[i].tick_params(axis='x', rotation=45)
        
        # Consistent Y-axis for Inter-note difference
        if 'Inter_note_difference' in feature:
            axes[i].set_ylim(-0.1, 2.5)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_features_boxplots.png'))
    plt.savefig(os.path.join(output_dir, 'top_features_boxplots.svg'))
    plt.close()

def plot_projection_complex(X_proj, y, loadings, feature_names, title, x_label, y_label, save_path, 
                            location_order, location_colors, wspace=0.05, legend_loc='upper right', xlim=None, ylim=None):
    fig = plt.figure(figsize=(18, 8))
    
    gs = fig.add_gridspec(2, 4, width_ratios=[4, 1, 0.5, 3], height_ratios=[4, 1], wspace=wspace, hspace=0.05)
    
    ax_main = fig.add_subplot(gs[0, 0])
    ax_hist_x = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_hist_y = fig.add_subplot(gs[0, 1], sharey=ax_main)
    ax_loadings = fig.add_subplot(gs[:, 3]) 
    
    # --- Main Scatter ---
    sns.scatterplot(x=X_proj[:, 0], y=X_proj[:, 1], hue=y, hue_order=location_order, 
                    palette=location_colors, s=30, edgecolor='none', alpha=0.7, ax=ax_main, legend=False) 
    ax_main.set_ylabel(y_label)
    ax_main.tick_params(labelbottom=False)
    ax_main.grid(False)
    
    # User Requested Limits
    if xlim:
        ax_main.set_xlim(xlim)
    if ylim:
        ax_main.set_ylim(ylim)
        
    # --- Marginal X (Bottom) ---
    for location in location_order:
        subset = X_proj[y == location, 0]
        sns.kdeplot(subset, color=location_colors[location], fill=True, alpha=0.3, ax=ax_hist_x, legend=False)
    ax_hist_x.set_xlabel(x_label)
    ax_hist_x.set_yticks([])
    ax_hist_x.grid(False)
    sns.despine(ax=ax_hist_x, left=True)

    # --- Marginal Y (Right) ---
    for location in location_order:
        subset = X_proj[y == location, 1]
        sns.kdeplot(y=subset, color=location_colors[location], fill=True, alpha=0.3, ax=ax_hist_y, legend=False)
    ax_hist_y.set_xlabel("")
    ax_hist_y.set_xticks([])
    ax_hist_y.tick_params(labelleft=False)
    ax_hist_y.grid(False)
    sns.despine(ax=ax_hist_y, bottom=True)
    
    # --- Loadings (Far Right) ---
    if loadings is not None:
        y_pos = np.arange(len(feature_names))
        ax_loadings.barh(y_pos, loadings[:, 0], align='center', label='Comp 1', alpha=0.7, color='gray')
        ax_loadings.barh(y_pos, loadings[:, 1], align='center', label='Comp 2', alpha=0.7, color='black')
        ax_loadings.set_yticks(y_pos)
        ax_loadings.set_yticklabels(feature_names)
        ax_loadings.set_xlabel("Feature Contribution")
        ax_loadings.set_title("Feature Loadings")
        ax_loadings.legend()
        ax_loadings.grid(False)
    else:
        ax_loadings.axis('off')
    
    # Overall Title
    fig.suptitle(title, fontsize=16)
    
    # Manual Legend for Locations
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=loc, markerfacecolor=location_colors[loc], markersize=10) for loc in location_order]
    
    ax_main.legend(handles=legend_elements, loc=legend_loc, title='Location')
    
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.png', '.svg'))
    plt.close()

def perform_statistics(df, features, target_col, output_dir):
    stats_results = []
    print("\n--- Statistical Analysis (ANOVA) ---")
    for feature in features:
        groups = [group[feature].values for name, group in df.groupby(target_col)]
        f_stat, p_val = stats.f_oneway(*groups)
        tukey = pairwise_tukeyhsd(endog=df[feature], groups=df[target_col], alpha=0.05)
        
        stats_results.append({
            'Feature': feature,
            'ANOVA F-stat': f_stat,
            'ANOVA p-val': p_val,
            'Significant': p_val < 0.05
        })
        
        with open(os.path.join(output_dir, 'stats_tukey_results.txt'), 'a') as f:
            f.write(f"\n\nVariable: {feature}\n")
            f.write(f"ANOVA F={f_stat:.2f}, p={p_val:.4e}\n")
            f.write(str(tukey.summary()))
            
    stats_df = pd.DataFrame(stats_results)
    stats_df.to_csv(os.path.join(output_dir, 'anova_results.csv'), index=False)
    print(stats_df)
    return stats_df

# --- Main Execution ---

print(f"Loading Data for {SPECIES_NAME} Geographic Variation...")
df_folder = os.path.join(PROJECT_PATH, 'dataframes')
master_df = pd.read_csv(os.path.join(df_folder, 'master_df.csv'))

# Filter for species
species_df = master_df[master_df['Species'] == SPECIES_NAME].copy()

# Filter locations with sufficient files
file_counts = species_df.groupby('Location')['File_name'].nunique()
valid_locations = file_counts[file_counts >= MIN_FILES_PER_LOCATION].index.tolist()
species_df = species_df[species_df['Location'].isin(valid_locations)]

# Filter out negative inter-note differences
if 'Inter_note_difference (s)' in species_df.columns:
    species_df = species_df[species_df['Inter_note_difference (s)'] >= 0]

y = species_df['Location']
X_all = species_df[acoustic_features]

# --- Validate Locations ---
available_locs = set(y.unique())
LOCATION_ORDER = [loc for loc in LOCATION_ORDER if loc in available_locs]

save_dir = os.path.join(PROJECT_PATH, 'Figures', 'geo_var_palmarum')
clear_output_dir(save_dir)

if X_all.isna().any().any():
    valid_indices = X_all.dropna().index
    X_all = X_all.loc[valid_indices]
    y = y.loc[valid_indices]
    species_df = species_df.loc[valid_indices]

scaler = StandardScaler()
X_scaled_all = pd.DataFrame(scaler.fit_transform(X_all), columns=X_all.columns, index=X_all.index)

# --- Feature Selection & Model Evaluation ---
print("\n--- Model Evaluation (80/20 Split) ---")
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X_scaled_all, y, test_size=0.2, random_state=SEED, stratify=y)

rf_eval = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, class_weight='balanced')
rf_eval.fit(X_train, y_train)
y_pred = rf_eval.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Random Forest Test Accuracy: {accuracy:.4f}")
with open(os.path.join(save_dir, 'rf_classification_report.txt'), 'w') as f:
    f.write(f"Random Forest Test Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

print("\n--- Feature Importance (Full Dataset) ---")
rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, class_weight='balanced')
rf.fit(X_scaled_all, y)

result = permutation_importance(rf, X_scaled_all, y, n_repeats=10, random_state=SEED, n_jobs=-1)
sorted_idx = result.importances_mean.argsort()[::-1]

feature_imp_df = pd.DataFrame({
    'Feature': X_all.columns[sorted_idx],
    'Importance_Mean': result.importances_mean[sorted_idx],
    'Importance_Std': result.importances_std[sorted_idx]
})

print("Top Features:")
print(feature_imp_df.head(10))

# 1. Feature Importance Plot
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance_Mean', y='Feature', data=feature_imp_df, xerr=feature_imp_df['Importance_Std'], color='#4c72b0')
plt.title(f'Random Forest Permutation Importance - {SPECIES_NAME}')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'permutation_importance_rf.png'))
plt.savefig(os.path.join(save_dir, 'permutation_importance_rf.svg'))
plt.close()

# --- Feature Subset Evaluation ---
sorted_features = feature_imp_df['Feature'].tolist()
n_features_list = range(1, len(sorted_features) + 1)
accuracy_scores = []
f1_scores = []

for k in n_features_list:
    current_features = sorted_features[:k]
    X_subset = X_scaled_all[current_features]
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_subset, y, test_size=0.2, random_state=SEED, stratify=y)
    
    rf_k = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, class_weight='balanced')
    rf_k.fit(X_tr, y_tr)
    y_pr = rf_k.predict(X_te)
    
    acc = accuracy_score(y_te, y_pr)
    f1 = f1_score(y_te, y_pr, average='weighted')
    
    accuracy_scores.append(acc)
    f1_scores.append(f1)

# 2. Elbow Plot
plt.figure(figsize=(10, 6))
plt.plot(n_features_list, accuracy_scores, marker='o', linestyle='-', color='#1f77b4', label='Accuracy')
plt.plot(n_features_list, f1_scores, marker='s', linestyle='--', color='#ff7f0e', label='F1 Score (Weighted)')
plt.title(f'Random Forest Feature Selection - {SPECIES_NAME}') 
plt.xlabel('#Features') 
plt.ylabel('Model Performance')
plt.xticks(n_features_list)
plt.legend(loc='lower right')
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_performance.png'))
plt.savefig(os.path.join(save_dir, 'feature_selection_performance.svg'))
plt.close()

# Select Top 6 Features
top_6_features = sorted_features[:6]
print(f"\nSelected Top 6 Features: {top_6_features}")
X_top6 = X_scaled_all[top_6_features]

# --- 3. LDA (User Limits: X -7 to 7, Y -5 to 5) ---
print("\nGenerating LDA Projection...")
lda = LinearDiscriminantAnalysis(n_components=min(2, len(LOCATION_ORDER)-1))
X_lda = lda.fit_transform(X_top6, y)
loadings_lda = lda.scalings_[:, :min(2, len(LOCATION_ORDER)-1)]

if X_lda.shape[1] == 2:
    plot_projection_complex(
        X_lda, y, loadings_lda, top_6_features, 
        f"LDA Projection - {SPECIES_NAME}",
        f"LD1 ({lda.explained_variance_ratio_[0]*100:.1f}%)",
        f"LD2 ({lda.explained_variance_ratio_[1]*100:.1f}%)",
        os.path.join(save_dir, 'lda_top_6.png'),
        LOCATION_ORDER, LOCATION_COLORS,
        wspace=0.3,
        legend_loc='upper left',
        xlim=(-7, 7), ylim=(-5, 5)
    )
    print(f"Saved LDA projection")

# --- 4. PCA (User Limits: X -5 to 5, Y -5 to 5) ---
print("\nGenerating PCA Projection...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_top6)
loadings_pca = pca.components_.T * np.sqrt(pca.explained_variance_)

plot_projection_complex(
    X_pca, y, loadings_pca, top_6_features,
    f"PCA Projection - {SPECIES_NAME}",
    f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
    f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
    os.path.join(save_dir, 'pca_top_6.png'),
    LOCATION_ORDER, LOCATION_COLORS,
    wspace=0.05,
    legend_loc='upper right',
    xlim=(-5, 5), ylim=(-5, 5)
)
print(f"Saved PCA projection")

# --- 5. Unsupervised UMAP ---
print("\nGenerating Unsupervised UMAP...")
try:
    import umap
    
    # UNSUPERVISED UMAP: do not pass y to fit_transform
    # increased n_neighbors for global structure
    
    umap_emb = umap.UMAP(n_neighbors=50, min_dist=0.5, n_components=2, random_state=SEED).fit_transform(X_top6)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=umap_emb[:, 0], y=umap_emb[:, 1], hue=y, hue_order=LOCATION_ORDER,
                    palette=LOCATION_COLORS, s=30, edgecolor='none', alpha=0.7)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(f"Unsupervised UMAP - {SPECIES_NAME}")
    plt.xticks([])
    plt.yticks([])
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'umap_top_6_features.png'))
    plt.savefig(os.path.join(save_dir, 'umap_top_6_features.svg'))
    plt.close()
    print(f"Saved UMAP projection")
except Exception as e:
    print(f"UMAP failed: {e}")

# --- 6. Boxplots & Statistics ---
print(f"\nGenerating Boxplots and Stats for Top 6 Features: {top_6_features}")
plot_feature_boxplots(species_df, top_6_features, 'Location', save_dir, LOCATION_ORDER, LOCATION_COLORS)
perform_statistics(species_df, top_6_features, 'Location', save_dir)

print(f"\n=== F04A {SPECIES_NAME} Analysis Complete ===")
