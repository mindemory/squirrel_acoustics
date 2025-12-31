
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
from sklearn.metrics import silhouette_score, f1_score
from sklearn.decomposition import PCA

from params import PROJECT_PATH, acoustic_features
from dimensionality_analysis import pca_analysis, umap_analysis

# --- Configuration ---
SPECIES_COLORS = {
    'F. palmarum': '#98003F',
    'F. tristriatus': '#5D4EA0',
    'F. pennanti': 'orange',
    'F. sublineatus': '#55ab0f'
}
SPECIES_ORDER = ['F. palmarum', 'F. tristriatus', 'F. pennanti', 'F. sublineatus']

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

def plot_feature_boxplots(df, features, target_col, output_dir):
    from itertools import combinations
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
        sns.boxplot(x=target_col, y=feature, data=df_plot, ax=axes[i], order=SPECIES_ORDER, palette=SPECIES_COLORS, showfliers=False)
        means = df_plot.groupby(target_col)[feature].mean()
        # Ensure means align with order
        axes[i].plot(range(len(SPECIES_ORDER)), [means[s] for s in SPECIES_ORDER], 
                     marker='D', color='white', markeredgecolor='black', linestyle='none', markersize=7, label='Mean')

        axes[i].set_title(feature)
        axes[i].tick_params(axis='x', rotation=45)
        
        if 'Inter_note_difference' in feature:
            axes[i].set_ylim(-0.1, 2.5) # Updated Range per user request
        
        # --- Add Pairwise Stats ---
        pairs = list(combinations(SPECIES_ORDER, 2))
        significant_pairs = []
        
        # Get data ranges for plotting lines
        # Using the plotting dataframe
        # y_max = df_plot[feature].quantile(0.99) 
        
        for s1, s2 in pairs:
            # Stats on PLOTTED data (kHz) - p-values identical to Hz, but just to be safe
            d1 = df_plot[df_plot[target_col] == s1][feature]
            d2 = df_plot[df_plot[target_col] == s2][feature]
            try:
                stat, p = stats.ttest_ind(d1, d2, nan_policy='omit')
                p_corr = p * 6
                if p_corr < 0.05:
                    star = "*"
                    if p_corr < 0.01: star = "**"
                    if p_corr < 0.001: star = "***"
                    significant_pairs.append(((s1, s2), star))
            except:
                pass

        # Plot bars for significant pairs
        y_lim_top = axes[i].get_ylim()[1]
        y_offset = (y_lim_top - axes[i].get_ylim()[0]) * 0.05
        current_y = y_lim_top
        
        axes[i].set_ylim(top=y_lim_top + (len(significant_pairs) * y_offset * 1.5))
        
        # Map species to x-indices
        x_map = {s: idx for idx, s in enumerate(SPECIES_ORDER)}
        
        for (s1, s2), label in significant_pairs:
            x1, x2 = x_map[s1], x_map[s2]
            current_y += y_offset
            h = y_offset * 0.2
            axes[i].plot([x1, x1, x2, x2], [current_y, current_y+h, current_y+h, current_y], lw=1, c='k')
            axes[i].text((x1+x2)*.5, current_y+h, label, ha='center', va='bottom', color='k', fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_features_boxplots.png'))
    plt.savefig(os.path.join(output_dir, 'top_features_boxplots.svg')) # Save SVG
    plt.close()

def plot_projection_complex(X_proj, y, loadings, feature_names, title, x_label, y_label, save_path, 
                            wspace=0.05, legend_loc='upper right', ylim=None):
    """
    Creates a complex figure with:
    - Main Scatter Plot (Top Left)
    - Marginal Histogram X (Bottom Left)
    - Marginal Histogram Y (Top Right)
    - Loadings Bar Plot (Far Right Panel)
    """
    fig = plt.figure(figsize=(18, 8)) # Wide figure to accommodate loadings
    
    # GridSpec with custom wspace
    gs = fig.add_gridspec(2, 4, width_ratios=[4, 1, 0.5, 3], height_ratios=[4, 1], wspace=wspace, hspace=0.05)
    
    ax_main = fig.add_subplot(gs[0, 0])
    ax_hist_x = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_hist_y = fig.add_subplot(gs[0, 1], sharey=ax_main)
    ax_loadings = fig.add_subplot(gs[:, 3]) 
    
    # --- Main Scatter ---
    sns.scatterplot(x=X_proj[:, 0], y=X_proj[:, 1], hue=y, hue_order=SPECIES_ORDER, 
                    palette=SPECIES_COLORS, s=30, edgecolor='none', alpha=0.7, ax=ax_main, legend=False) 
    ax_main.set_ylabel(y_label)
    ax_main.tick_params(labelbottom=False)
    ax_main.grid(False)
    
    if ylim:
        ax_main.set_ylim(ylim)
        
    # --- Marginal X (Bottom) ---
    for species in SPECIES_ORDER:
        subset = X_proj[y == species, 0]
        sns.kdeplot(subset, color=SPECIES_COLORS[species], fill=True, alpha=0.3, ax=ax_hist_x, legend=False)
    ax_hist_x.set_xlabel(x_label)
    ax_hist_x.set_yticks([])
    ax_hist_x.grid(False)
    sns.despine(ax=ax_hist_x, left=True)

    # --- Marginal Y (Right) ---
    for species in SPECIES_ORDER:
        subset = X_proj[y == species, 1]
        sns.kdeplot(y=subset, color=SPECIES_COLORS[species], fill=True, alpha=0.3, ax=ax_hist_y, legend=False)
    ax_hist_y.set_xlabel("")
    ax_hist_y.set_xticks([])
    ax_hist_y.tick_params(labelleft=False)
    ax_hist_y.grid(False)
    sns.despine(ax=ax_hist_y, bottom=True)
    
    # --- Loadings (Far Right) ---
    y_pos = np.arange(len(feature_names))
    ax_loadings.barh(y_pos, loadings[:, 0], align='center', label='Comp 1', alpha=0.7, color='gray')
    ax_loadings.barh(y_pos, loadings[:, 1], align='center', label='Comp 2', alpha=0.7, color='black')
    ax_loadings.set_yticks(y_pos)
    ax_loadings.set_yticklabels(feature_names)
    ax_loadings.set_xlabel("Feature Contribution")
    ax_loadings.set_title("Feature Loadings")
    ax_loadings.legend()
    ax_loadings.grid(False) # No grid
    
    # Overall Title
    fig.suptitle(title, fontsize=16)
    
    # Manual Legend for Species
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=s, markerfacecolor=SPECIES_COLORS[s], markersize=10) for s in SPECIES_ORDER]
    
    # Legend location customizable
    ax_main.legend(handles=legend_elements, loc=legend_loc, title='Species')
    
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.png', '.svg')) # Save SVG
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

print("Loading Data...")
df_folder = os.path.join(PROJECT_PATH, 'dataframes')
# User requested to use master_random_df (Main Analysis)
master_random_df = pd.read_csv(os.path.join(df_folder, 'master_random_df.csv'))
y = master_random_df['Species']

# Filter out negative inter-note differences (overlapping notes)
if 'Inter_note_difference (s)' in master_random_df.columns:
    print("Filtering negative inter-note differences...")
    before_count = len(master_random_df)
    master_random_df = master_random_df[master_random_df['Inter_note_difference (s)'] >= 0]
    print(f"Removed {before_count - len(master_random_df)} rows with negative intervals.")
    y = master_random_df['Species']

X_all = master_random_df[acoustic_features]

save_dir = os.path.join(PROJECT_PATH, 'Figures', 'dimred')
print(f"Clearing output directory: {save_dir}")
clear_output_dir(save_dir)

if X_all.isna().any().any():
    print("Warning: NaNs found. Dropping rows with NaNs.")
    valid_indices = X_all.dropna().index
    X_all = X_all.loc[valid_indices]
    y = y.loc[valid_indices]
    master_random_df = master_random_df.loc[valid_indices]

scaler = StandardScaler()
X_scaled_all = pd.DataFrame(scaler.fit_transform(X_all), columns=X_all.columns, index=X_all.index)

# --- Feature Selection & Model Evaluation ---
print("\n--- Model Evaluation (80/20 Split) ---")
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X_scaled_all, y, test_size=0.2, random_state=42, stratify=y)

rf_eval = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf_eval.fit(X_train, y_train)
y_pred = rf_eval.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Random Forest Test Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# Save report
with open(os.path.join(save_dir, 'rf_classification_report.txt'), 'w') as f:
    f.write(f"Random Forest Test Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

print("\n--- Feature Importance (Full Dataset) ---")
print("Retraining on all data for feature selection...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_scaled_all, y)

result = permutation_importance(rf, X_scaled_all, y, n_repeats=10, random_state=42, n_jobs=-1)
sorted_idx = result.importances_mean.argsort()[::-1]

feature_imp_df = pd.DataFrame({
    'Feature': X_all.columns[sorted_idx],
    'Importance_Mean': result.importances_mean[sorted_idx],
    'Importance_Std': result.importances_std[sorted_idx]
})

print("Top Features:")
print(feature_imp_df.head(10))

plt.figure(figsize=(12, 8))
# 3A Updates: Mono color (e.g., steelblue or gray), remove palette
sns.barplot(x='Importance_Mean', y='Feature', data=feature_imp_df, xerr=feature_imp_df['Importance_Std'], color='#4c72b0')
plt.title('Random Forest Classifier Permutation Importance')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'permutation_importance_rf.png'))
plt.savefig(os.path.join(save_dir, 'permutation_importance_rf.svg'))
plt.close()

# --- Feature Subset Evaluation (Silhouette Score Analysis) ---
# --- Classification Performance Analysis (Accuracy & F1 vs Features) ---
print("\n--- Feature Subset Evaluation (Classification Performance) ---")
# Evaluate Top k features for k = 1 to len(all_features)
sorted_features = feature_imp_df['Feature'].tolist()
n_features_list = range(1, len(sorted_features) + 1)
accuracy_scores = []
f1_scores = []

print("Calculating Accuracy and F1 Scores for Top-k feature subsets...")

for k in n_features_list:
    current_features = sorted_features[:k]
    X_subset = X_scaled_all[current_features]
    
    # Split data (ensure consistency with main evaluation)
    X_tr, X_te, y_tr, y_te = train_test_split(X_subset, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train RF
    rf_k = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf_k.fit(X_tr, y_tr)
    y_pr = rf_k.predict(X_te)
    
    acc = accuracy_score(y_te, y_pr)
    f1 = f1_score(y_te, y_pr, average='weighted')
    
    accuracy_scores.append(acc)
    f1_scores.append(f1)
    
    print(f"Top {k} features: Accuracy={acc:.4f}, F1={f1:.4f}")

# Plot Figure 3B: Feature Selection Performance
plt.figure(figsize=(10, 6))
plt.plot(n_features_list, accuracy_scores, marker='o', linestyle='-', color='#1f77b4', label='Accuracy')
plt.plot(n_features_list, f1_scores, marker='s', linestyle='--', color='#ff7f0e', label='F1 Score (Weighted)')
plt.title('Random Forest Feature Selection') 
plt.xlabel('#Features') 
plt.ylabel('Model Performance') # Updated Label
plt.xticks(n_features_list)

# 3B Y-axis 0.5 to 1.0 with ticks
plt.ylim(0.5, 1.0)
plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# Highlight Accuracy at 6 (Text Below Line)
acc_at_6 = accuracy_scores[5] 
# Position text below: y - 0.08, arrow styles
plt.annotate(f'{acc_at_6:.4f}', xy=(6, acc_at_6), xytext=(6, acc_at_6 - 0.08),
             arrowprops=dict(facecolor='black', shrink=0.05), ha='center')

plt.axvline(x=6, color='r', linestyle=':', linewidth=2, label='Selected (6 Features)')
plt.legend(loc='lower right')
plt.grid(False) # No grid
plt.tight_layout()
filename_3b = 'feature_selection_performance.png'
plt.savefig(os.path.join(save_dir, filename_3b))
plt.savefig(os.path.join(save_dir, filename_3b.replace('.png', '.svg')))
plt.close()
print(f"Saved Figure 3B to: {filename_3b}")

# Select Top 6 Features
top_6_features = sorted_features[:6]
print(f"\nSelected Top 6 Features: {top_6_features}")
X_top6 = X_scaled_all[top_6_features]

# --- LDA (Figure 3C - SWAPPED) ---
print("\nGenerating Figure 3C (LDA + Marginals + Loadings)...")
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_top6, y)
loadings_lda = lda.scalings_[:, :2]

plot_projection_complex(
    X_lda, y, loadings_lda, top_6_features, 
    "LDA Projection (Top 6 Features)",
    f"LD1 ({lda.explained_variance_ratio_[0]*100:.1f}%)",
    f"LD2 ({lda.explained_variance_ratio_[1]*100:.1f}%)",
    os.path.join(save_dir, 'lda_top_6.png'),
    wspace=0.3, # Increased space between plots (User 3C request)
    legend_loc='upper left' # User 3C request
)
print(f"Saved Figure 3C to: lda_top_6.png") # User requested LDA as 3C

# --- PCA (Figure 3D - SWAPPED) ---
print("\nGenerating Figure 3D (PCA + Marginals + Loadings)...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_top6)
loadings_pca = pca.components_.T * np.sqrt(pca.explained_variance_)

plot_projection_complex(
    X_pca, y, loadings_pca, top_6_features,
    "PCA Projection (Top 6 Features)",
    f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
    f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
    os.path.join(save_dir, 'pca_top_6.png'),
    wspace=0.05, # Default tight
    ylim=(-2, 7.5), # User 3D request (-2, 7.5)
    legend_loc='upper right'
)
print(f"Saved Figure 3D to: pca_top_6.png") # User requested PCA as 3D


# --- UMAP (Figure 3E) ---
print("\nGenerating Figure 3E (Supervised UMAP with Top 6 Features)...")
try:
    import umap
    # SUPERVISED UMAP: pass y to fit_transform
    # increased n_neighbors for global structure, decreased min_dist for tight clusters
    
    # Encode target to numeric for UMAP
    y_encoded = pd.factorize(y)[0]
    
    # 3E User Request: n_neighbors=50
    umap_emb = umap.UMAP(n_neighbors=50, min_dist=0.5, n_components=2, random_state=42).fit_transform(X_top6, y=y_encoded)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=umap_emb[:, 0], y=umap_emb[:, 1], hue=y, hue_order=SPECIES_ORDER,
                    palette=SPECIES_COLORS, s=30, edgecolor='none', alpha=0.7)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(f"Supervised UMAP Projection (Top 6 Features)")
    # Remove ticks/spines for "nicer" look
    plt.xticks([])
    plt.yticks([])
    sns.despine() # Remove box
    plt.tight_layout()
    filename_3e = 'umap_top_6_features.png'
    plt.savefig(os.path.join(save_dir, filename_3e))
    plt.savefig(os.path.join(save_dir, filename_3e.replace('.png', '.svg')))
    plt.close()
    print(f"Saved Figure 3E to: {filename_3e}")
except Exception as e:
    print(f"UMAP failed: {e}")


# --- Boxplots & Statistics (Figure 3F) ---
print(f"\nGenerating Boxplots and Stats for Top 6 Features: {top_6_features}")
# This generates Figure 3F
# Note: Existing function handles 6 features gracefully (2 rows of 3)
plot_feature_boxplots(master_random_df, top_6_features, 'Species', save_dir)
perform_statistics(master_random_df, top_6_features, 'Species', save_dir)

