
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from params import PROJECT_PATH

# --- Configuration ---
OUTPUT_DIR = os.path.join(PROJECT_PATH, 'Figures', 'acoustic_adaptation')

# Target Species and Colors
TARGET_SPECIES = ['F. palmarum', 'F. tristriatus', 'F. pennanti']
SPECIES_COLORS = {
    'F. palmarum': '#98003F',
    'F. tristriatus': '#5D4EA0',
    'F. pennanti': 'orange'
}

# Acoustic Features (Rows)
ACOUSTIC_FEATURES = [
    'PFC Max Freq (Hz)', 
    'PFC Min Freq (Hz)', 
    'BW 90% (Hz)', 
    'Agg Entropy (bits)'
]

# Environmental Variables (One Figure per Variable)
# These names must match locations_dec.csv
ENV_VARIABLES = [
    # Natural
    'forest_canopy_19(metres)',
    'NDVI_19_20',
    'Tmean_18_19 (C)',
    'Pmean_17_18_19(mm)',
    'Tmax_18_19(C)',
    'Tmin_18_19(C)',
    'Temperature seasonality (mm)',
    'STDDEV',
    'mean',
    # Anthropogenic
    'Human_pop_density_2019_100m',
    'Human_pop_density_2020_100m',
    'Built_settlement_2019',
    'Built_settlement_2020',
    'Nightlight_2019'
]

# --- Helper Functions ---
def clear_output_dir(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(directory)

def clean_filename(fname):
    """Normalize filenames for merging (remove path, extension, lower case)"""
    base = os.path.basename(str(fname))
    root, ext = os.path.splitext(base)
    return root.lower()

# --- Main Execution ---
print("--- F07: Acoustic Adaptation Hypothesis Analysis (Refactored) ---")

# 1. Load Data
df_folder = os.path.join(PROJECT_PATH, 'dataframes')
print("Loading master_df.csv...")
master_df = pd.read_csv(os.path.join(df_folder, 'master_df.csv'))

print("Loading locations_dec.csv...")
loc_df = pd.read_csv(os.path.join(PROJECT_PATH, 'locations_dec.csv'))


# 2. Prepare Acoustic Data
print("Filtering and Aggregating Acoustic Data...")
# Filter for target species
df_filtered = master_df[master_df['Species'].isin(TARGET_SPECIES)].copy()

# Aggregate by Audio File (Begin File)
# We need Mean of Acoustic Features for each file
# Also keep Species column (should be constant per file)
acoustic_means = df_filtered.groupby(['Begin File', 'Species'])[ACOUSTIC_FEATURES].mean().reset_index()

# 3. Prepare Location Data
# Create 'merge_key' in both dataframes
# master_df uses 'Begin File' for audio name
acoustic_means['merge_key'] = acoustic_means['Begin File'].apply(clean_filename)
loc_df['merge_key'] = loc_df['12_Audio_file_name'].apply(clean_filename)

# Select relevant columns from loc_df (merge_key + env vars)
# Check for duplicates in loc_df (one row per file expected)
loc_subset = loc_df[['merge_key'] + ENV_VARIABLES].drop_duplicates(subset=['merge_key'])

# 4. Merge
print("Merging Acoustic and Environmental Data...")
# Show sample keys before merge for debugging
print(f"Sample Acoustic Keys: {acoustic_means['merge_key'].head().tolist()}")
print(f"Sample Location Keys: {loc_subset['merge_key'].head().tolist()}")

merged_df = pd.merge(acoustic_means, loc_subset, on='merge_key', how='inner')
print(f"Merged: {len(merged_df)} files (from {len(acoustic_means)} acoustic files)")


if len(merged_df) == 0:
    print("CRITICAL ERROR: No files matched. Aborting.")
    exit(1)

# 5. Generate Figures
clear_output_dir(OUTPUT_DIR)
print(f"Saving outputs to: {OUTPUT_DIR}")

for env_var in ENV_VARIABLES:
    if env_var not in merged_df.columns:
        print(f"Skipping missing variable: {env_var}")
        continue
        
    print(f"Generating Figure for: {env_var}")
    
    # Create 4x3 Grid
    # Rows: Features, Cols: Species
    fig, axes = plt.subplots(4, 3, figsize=(15, 16), sharex=False, sharey=False) # Share axes? No, ranges differ.
    
    # Adjust spacing
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Main Title
    fig.suptitle(f"Variation with {env_var}", fontsize=20, y=0.98)
    
    # Headers for Columns (Species)
    for col, species in enumerate(TARGET_SPECIES):
        axes[0, col].set_title(species, fontsize=16, pad=10, color=SPECIES_COLORS[species], fontweight='bold')
        
    # Process Grid
    for row, feature in enumerate(ACOUSTIC_FEATURES):
        # Set Row Label (Feature) - tricky placement, maybe just ylabel on first col
        
        for col, species in enumerate(TARGET_SPECIES):
            ax = axes[row, col]
            
            # Filter Data
            subset = merged_df[merged_df['Species'] == species]
            
            # Drop NaNs
            valid_data = subset[[env_var, feature]].dropna()
            
            x = valid_data[env_var]
            y = valid_data[feature]
            
            # Formatting
            ax.set_facecolor('white')
            ax.grid(False) # REQUIRED: Remove grid
            
            if len(x) < 3:
                ax.text(0.5, 0.5, "n < 3", ha='center', va='center')
                continue
                
            # Scatter Plot
            ax.scatter(x, y, color=SPECIES_COLORS[species], alpha=0.6, s=40, edgecolors='white', linewidth=0.5)
            
            # Regression Fit
            if len(x.unique()) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                line = slope * x + intercept
                ax.plot(x, line, color='black', alpha=0.7, linestyle='--', linewidth=1.5)
                
                # Annotated Stats (r, p)
                # Position: Top Right or Top Left?
                stat_str = f"r={r_value:.2f}\np={p_value:.3f}"
                ax.annotate(stat_str, xy=(0.05, 0.95), xycoords='axes fraction', 
                            fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))
            
            # Labels
            if col == 0:
                ax.set_ylabel(feature, fontsize=11, fontweight='bold')
            else:
                ax.set_ylabel("")
                
            if row == 3:
                ax.set_xlabel(env_var, fontsize=10)
            else:
                ax.set_xlabel("")
                
            # Aesthetic Spines
            sns.despine(ax=ax)
            
    # Save Figure
    safe_var = env_var.replace('(', '').replace(')', '').replace(' ', '_').replace('/', '_')
    save_path = os.path.join(OUTPUT_DIR, f"{safe_var}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.close()

print("F07 Analysis Complete.")
