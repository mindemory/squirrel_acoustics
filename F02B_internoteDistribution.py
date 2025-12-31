import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import anderson_ksamp, ks_2samp
from itertools import combinations
from params import PROJECT_PATH, species_list, sub_bout_difference_dict_F, bins_dict, bout_difference_dict

# Load master_df
df_folder = os.path.join(PROJECT_PATH, 'dataframes')
master_df = pd.read_csv(os.path.join(df_folder, 'master_df.csv'))

# Create master_df_bout_version by removing first row from each file
master_df_bout_version = pd.DataFrame()
for bf in master_df['File_name'].unique():
    temp_df = master_df[master_df['File_name'] == bf].reset_index(drop=True)
    temp_df = temp_df.drop([0]).reset_index(drop=True)
    master_df_bout_version = pd.concat([master_df_bout_version, temp_df])

# Calculate statistics for each species
stats_list = []
for spp in master_df_bout_version['Species'].unique():
    # Filter for intervals > sub-bout threshold (these are the potential bouts)
    bout_df_temp = master_df_bout_version[master_df_bout_version['Species'] == spp]
    bout_df_temp = bout_df_temp[bout_df_temp['Inter_note_difference (s)'] > sub_bout_difference_dict_F[spp]]
    
    # Calculate Stats
    min_val = bout_df_temp['Inter_note_difference (s)'].min()
    Q1, M, Q3 = bout_df_temp['Inter_note_difference (s)'].quantile([0.25, 0.5, 0.75])
    IQR = Q3 - Q1
    LB = Q1 - 1.5 * IQR
    UB = Q3 + 1.5 * IQR
    
    print(f"{spp} Min: {min_val}")
    print(f"Q1: {Q1}, Median: {M}, Q3: {Q3}")
    print(f"IQR: {IQR}, LB: {LB}, UB: {UB}")
    print()
    
    stats_list.append({
        'Species': spp,
        'Sub_Bout_Threshold_Used': sub_bout_difference_dict_F[spp],
        'Min_Inter_Bout_Interval': min_val,
        'Q1': Q1,
        'Median': M,
        'Q3': Q3,
        'IQR': IQR,
        'Lower_Bound': LB,
        'Upper_Bound': UB,
        'Count': len(bout_df_temp)
    })

# Save stats to CSV
stats_df = pd.DataFrame(stats_list)
stats_path = os.path.join(PROJECT_PATH, 'dataframes', 'bout_distribution_metrics.csv')
stats_df.to_csv(stats_path, index=False)
print(f"Bout metrics saved to: {stats_path}")

# Create the inter-note distribution figure
# Standardize bin width across all subplots
x_min, x_max = -0.1, 1.0
bin_width = 0.01  # Standard bin width in seconds
bins = np.arange(x_min, x_max + bin_width, bin_width)

# Define species order: palmarum, tristriatus, pennanti, sublineatus
desired_species_order = ['palmarum', 'tristriatus', 'pennanti', 'sublineatus']

# --- Anderson-Darling k-sample Test ---
print("\n--- Anderson-Darling k-sample Test (Inter-note Intervals) ---")
# Collect arrays for each species
intervals_by_species = []
species_names_for_test = []

for spp in desired_species_order:
    vals = master_df_bout_version[master_df_bout_version['Species'] == 'F. ' + spp]['Inter_note_difference (s)'].dropna()
    intervals_by_species.append(vals.values)
    species_names_for_test.append('F. ' + spp)

try:
    # anderson_ksamp returns (statistic, critical_values, significance_level)
    # The significance_level is the p-value.
    # Note: scipy implementation might clamp p-value at 0.001 or 0.25 (interpolated).
    res = anderson_ksamp(intervals_by_species)
    print(f"Comparing species: {species_names_for_test}")
    print(f"AD Statistic: {res.statistic:.4f}")
    print(f"Significance Level (p-value): {res.significance_level:.5f}")
    
    if res.significance_level < 0.05:
        print("Result: SIGNIFICANT. The inter-note interval distributions differ significantly across species.")
    else:
        print("Result: NOT SIGNIFICANT.")
except Exception as e:
    print(f"Anderson-Darling Test Failed: {e}")

# --- Post-hoc Pairwise KS Tests ---
print("\n--- Post-hoc Pairwise Kolmogorov-Smirnov Tests ---")
pairs = list(combinations(range(len(species_names_for_test)), 2))
alpha = 0.05
bonferroni_alpha = alpha / len(pairs)
print(f"Bonferroni Corrected Alpha: {bonferroni_alpha:.5f} (Base alpha: {alpha} / {len(pairs)} pairs)")

results_ks = []
for i, j in pairs:
    spp1_name = species_names_for_test[i]
    spp2_name = species_names_for_test[j]
    data1 = intervals_by_species[i]
    data2 = intervals_by_species[j]
    
    stat, p_val = ks_2samp(data1, data2)
    significant = p_val < bonferroni_alpha
    
    results_ks.append({
        'Pair': f"{spp1_name} vs {spp2_name}",
        'KS Statistic': stat,
        'P-value': p_val,
        'Significant': significant
    })
    
    sig_str = "*" if significant else "ns"
    print(f"{spp1_name} vs {spp2_name}: KS={stat:.4f}, p={p_val:.4e} [{sig_str}]")

print("-" * 30)

fig, ax = plt.subplots(nrows=2, ncols=2, dpi=100, figsize=(12, 10), sharex=True, sharey=True)
fig, ax = plt.subplots(nrows=2, ncols=2, dpi=100, figsize=(12, 10), sharex=True, sharey=True)
ax = ax.flatten()  # Flatten to 1D array for easier indexing

# Plotting
for i in range(len(desired_species_order)):
    spp = desired_species_order[i]
    temp_df = master_df_bout_version[master_df_bout_version['Species'] == 'F. ' + spp]['Inter_note_difference (s)']
    temp_df.hist(ax=ax[i], density=1, bins=bins, xlabelsize=12, ylabelsize=11, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Add threshold lines
    sub_bout_thresh = sub_bout_difference_dict_F['F. ' + spp]
    bout_thresh = bout_difference_dict[spp]
    
    ax[i].axvline(sub_bout_thresh, color='red', linestyle='--', linewidth=1.5, label='Sub-bout Thresh')
    ax[i].axvline(bout_thresh, color='green', linestyle='-.', linewidth=1.5, label='Bout Thresh')
    
    ax[i].set_title('F. ' + spp, fontsize=14)
    # Set labels and ticks for all subplots
    ax[i].set_xlabel('Internote differences (s)', fontsize=12)
    ax[i].set_ylabel('Frequency (%)', fontsize=12)
    ax[i].set_xlim(x_min, x_max)
    ax[i].set_ylim(0, 15)
    ax[i].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax[i].tick_params(axis='both', which='major', labelsize=11)
    ax[i].grid(False)
    
    # Add legend to the first subplot or all
    ax[i].legend(loc='upper right', fontsize=8)

fig.subplots_adjust(wspace=0.3, hspace=0.3)

# Save the figure in both PNG and SVG formats
output_dir = os.path.join(PROJECT_PATH, 'Figures')
os.makedirs(output_dir, exist_ok=True)

# Save as PNG
png_path = os.path.join(output_dir, 'inter_note_distribution.png')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
print(f"Inter-note distribution saved as PNG to: {png_path}")

# Save as SVG (editable format)
svg_path = os.path.join(output_dir, 'inter_note_distribution.svg')
plt.savefig(svg_path, format='svg', bbox_inches='tight')
print(f"Inter-note distribution saved as SVG to: {svg_path}")

plt.close(fig)

