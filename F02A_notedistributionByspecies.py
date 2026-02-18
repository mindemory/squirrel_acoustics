import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from params import PROJECT_PATH, species_list, species_colors

# Load master_good_df
df_folder = os.path.join(PROJECT_PATH, 'dataframes')
master_good_df = pd.read_csv(os.path.join(df_folder, 'master_good_df.csv'))

# Create note distribution dataframe
df = pd.DataFrame()
for spp in species_list:
    Fspp = 'F. ' + spp
    spec_count = pd.DataFrame()
    spec_count[Fspp] = master_good_df[master_good_df['Species'] == Fspp]['Note'].value_counts()
    df = pd.concat([df, spec_count], axis=1)

df = df.fillna(0)
df = df.astype('int32')
df['Total'] = df.sum(axis=1)
df = df.sort_values(by='Total', axis=0, ascending=False)
# Keep Total for stats, then drop later or use row sums
row_sums = df['Total'] # This is sorted descending
grand_total = row_sums.sum()
num_unique_notes = len(df)

print("-" * 30)
print(f"Total Unique Note Types across all species: {num_unique_notes}")
print(f"Total Notes Analyzed: {grand_total}")
print("-" * 30)

# Top 10
top_10 = row_sums.iloc[:10]
top_10_sum = top_10.sum()
top_10_pct = (top_10_sum / grand_total) * 100
print(f"Top 10 Notes contribute: {top_10_pct:.2f}%")
for note, count in top_10.items():
    pct = (count / grand_total) * 100
    print(f"  {note}: {count} ({pct:.2f}%)")

# Middle 10 (11-20)
if num_unique_notes > 10:
    mid_10 = row_sums.iloc[10:20]
    mid_10_sum = mid_10.sum()
    mid_10_pct = (mid_10_sum / grand_total) * 100
    print("-" * 30)
    print(f"Middle 10 Notes (Rank 11-20) contribute: {mid_10_pct:.2f}%")
else:
    print("No Middle 10 notes.")

# Bottom (21+)
if num_unique_notes > 20:
    bottom = row_sums.iloc[20:]
    bottom_sum = bottom.sum()
    bottom_pct = (bottom_sum / grand_total) * 100
    print("-" * 30)
    print(f"Bottom {num_unique_notes - 20} Notes (Rank 21+) contribute: {bottom_pct:.2f}%")
else:
    print("No Bottom notes (Rank 21+).")
print("-" * 30)

df = df.drop(['Total'], axis=1)

# Reorder columns to match desired species order
desired_species_order = ['F. palmarum', 'F. tristriatus', 'F. pennanti', 'F. sublineatus', 'F. layardi', 'F. obscurus']
df = df.reindex(columns=[col for col in desired_species_order if col in df.columns])

# Create percentage dataframes
# Subplot 0: % of note count across all recordings (normalize by total across all species)
total_all_notes = df.sum().sum()
df_pct_all = (df / total_all_notes * 100).round(2)

# Subplot 1: Group notes - keep only specified notes, group others as "Other"
notes_to_keep = ['IU', 'IU-RD', 'lIU-RD', 'RD', 'IU-LD', 'lIU']
df_species_grouped = pd.DataFrame()

for col in df.columns:
    species_data = df[col].copy()
    # Create a new series with grouped notes
    grouped_data = pd.Series(dtype='float64')
    
    # Keep specified notes
    for note in notes_to_keep:
        if note in species_data.index:
            grouped_data[note] = species_data[note]
    
    # Sum all other notes as "Other"
    other_notes = [note for note in species_data.index if note not in notes_to_keep]
    if other_notes:
        other_sum = species_data[other_notes].sum()
        if other_sum > 0:
            grouped_data['Other'] = other_sum
    
    df_species_grouped[col] = grouped_data

df_species_grouped = df_species_grouped.fillna(0).astype('int32')

# Reorder columns to match desired species order
df_species_grouped = df_species_grouped.reindex(columns=[col for col in desired_species_order if col in df_species_grouped.columns])

# Calculate percentages for subplot 1 (normalize by each species total)
df_pct_species = df_species_grouped.copy()
for col in df_pct_species.columns:
    species_total = df_pct_species[col].sum()
    if species_total > 0:
        df_pct_species[col] = (df_pct_species[col] / species_total * 100).round(2)

# Define custom ordering for each species
species_orders = {
    'F. palmarum': ['IU', 'IU-RD', 'IU-LD', 'RD', 'lIU', 'lIU-RD', 'Other'],
    'F. tristriatus': ['IU-RD', 'lIU-RD', 'RD', 'IU', 'lIU', 'IU-LD', 'Other'],
    'F. pennanti': ['IU', 'lIU', 'IU-RD', 'IU-LD', 'RD', 'lIU-RD', 'Other'],
    'F. sublineatus': ['IU', 'RD', 'IU-LD', 'IU-RD', 'lIU', 'lIU-RD', 'Other'],
    'F. layardi': ['IU', 'RD', 'IU-LD', 'IU-RD', 'lIU', 'lIU-RD', 'Other'],
    'F. obscurus': ['IU', 'RD', 'IU-LD', 'IU-RD', 'lIU', 'lIU-RD', 'Other']
}

# Create the figure with two subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8), dpi=100)

# Plot 1: Note abundance and distribution across species (% of all recordings)
# Use same colors as F01, in order
species_colors_subplot0 = [species_colors[spp] for spp in desired_species_order if spp in df_pct_all.index or spp in df_pct_all.columns]
df_pct_all.plot.barh(ax=axes[0], color=species_colors_subplot0, stacked=True,
                     title='\nNote abundance and distribution across species\n', rot=0)
axes[0].set_xlabel('% of note count across all recordings')

# Plot 2: Note distribution within species (% of notes for each species)
# Define color mapping using Set2 palette (colorblind-safe)
note_color_map = {
    'IU': '#66c2a5',      # Teal
    'IU-RD': '#fc8d62',  # Coral
    'IU-LD': '#8da0cb',  # Blue
    'RD': '#e78ac3',     # Pink
    'lIU': '#a6d854',    # Lime green
    'lIU-RD': '#ffd92f', # Yellow
    'Other': '#e5c494'   # Tan
}

# Manually create stacked bars with custom ordering for each species
x_positions = np.arange(len(df_pct_species.columns))
bottom = np.zeros(len(df_pct_species.columns))

# Track which notes have been added to legend
notes_in_legend = set()

# Plot each species with its custom order
for i, species_col in enumerate(df_pct_species.columns):
    order = species_orders.get(species_col, [])
    current_bottom = 0
    
    # Stack notes in the specified order for this species
    for note in order:
        if note in df_pct_species.index:
            value = df_pct_species.loc[note, species_col]
            if value > 0:
                # Plot this segment for this species only
                axes[1].bar(i, value, bottom=current_bottom, 
                           color=note_color_map.get(note, 'gray'),
                           label=note if note not in notes_in_legend else '')
                notes_in_legend.add(note)
                current_bottom += value

# Set x-axis labels and positions
axes[1].set_xticks(x_positions)
axes[1].set_xticklabels(df_pct_species.columns, rotation=45, ha='right')
axes[1].set_ylabel('% of notes across all recordings for that species')
axes[1].set_title('\nNote distribution within species\n')
axes[1].legend(bbox_to_anchor=(1.25, 0.95))

axes[1].legend(bbox_to_anchor=(1.25, 0.95))

# --- Chi-Square Test for Independence (Top Note Types) ---
print("\n--- Chi-Square Test of Independence ---")
print(f"Testing distribution of Top Notes: {notes_to_keep} across {desired_species_order}")

# Filter for the top notes
# df_species_grouped has columns=Species, index=Notes (including 'Other')
# We want to test the specific notes we visualized.
# Use df_species_grouped but exclude 'Other' if we only want "top note types". 
# The user said "top note types".
# If we include "Other", it tests the whole distribution. 
# "compare the equal abundance of the top note types" -> Suggests focusing on those specific types.
# Let's use the counts from df (which has all notes) filtered by notes_to_keep.

contingency_table = df.loc[notes_to_keep].T # Transpose: Rows=Species, Cols=Notes
# Order Species according to desired_species_order
contingency_table = contingency_table.reindex(desired_species_order)
contingency_table = contingency_table.fillna(0)

print("\nContingency Table (Counts):")
print(contingency_table)

chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.100e}")
print(f"Degrees of Freedom: {dof}")

if p < 0.05:
    print("Result: SIGNIFICANT. The distribution of top note types varies significantly across species.")
else:
    print("Result: NOT SIGNIFICANT. Top note type distribution is consistent across species.")
print("-" * 30)


# Save the figure in both PNG and SVG formats
output_dir = os.path.join(PROJECT_PATH, 'Figures')
os.makedirs(output_dir, exist_ok=True)

# Save as PNG
png_path = os.path.join(output_dir, 'note_distribution.png')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
print(f"Note distribution saved as PNG to: {png_path}")

# Save as SVG (editable format)
svg_path = os.path.join(output_dir, 'note_distribution.svg')
plt.savefig(svg_path, format='svg', bbox_inches='tight')
print(f"Note distribution saved as SVG to: {svg_path}")

plt.close(fig)

# --- Additional Figure: Note Distribution with Split Palmarum (India vs Sri Lanka) ---
print("\n--- Generating Split Population Figure (India vs Sri Lanka) ---")

# Create a working copy for split analysis
df_split = master_good_df.copy()

# Update Species column for F. palmarum based on Location
# Note: 'Colombo' is the location for Sri Lanka
df_split.loc[(df_split['Species'] == 'F. palmarum') & (df_split['Location'] == 'Colombo'), 'Species'] = 'F. palmarum (Sri Lanka)'
df_split.loc[(df_split['Species'] == 'F. palmarum') & (df_split['Location'] != 'Colombo'), 'Species'] = 'F. palmarum (India)'

# Define new species list and colors
split_species_order = ['F. palmarum (India)', 'F. palmarum (Sri Lanka)', 'F. tristriatus', 'F. pennanti', 'F. sublineatus', 'F. layardi', 'F. obscurus']

# Create Note Counts for Split Data
df_counts_split = pd.DataFrame()
for spp_full in split_species_order:
    # Handle the fact that species_list in params doesn't have the split names
    # and master_good_df 'Species' column already has 'F. ' prefix
    spec_count = pd.DataFrame()
    spec_count[spp_full] = df_split[df_split['Species'] == spp_full]['Note'].value_counts()
    df_counts_split = pd.concat([df_counts_split, spec_count], axis=1)

df_counts_split = df_counts_split.fillna(0).astype('int32')

# Create percentage dataframes for Split Data
# Subplot 0: % of note count across all recordings
total_all_notes_split = df_counts_split.sum().sum()
df_pct_all_split = (df_counts_split / total_all_notes_split * 100).round(2)

# Subplot 1: Group notes
df_species_grouped_split = pd.DataFrame()
for col in df_counts_split.columns:
    species_data = df_counts_split[col].copy()
    grouped_data = pd.Series(dtype='float64')
    
    for note in notes_to_keep:
        if note in species_data.index:
            grouped_data[note] = species_data[note]
    
    other_notes = [note for note in species_data.index if note not in notes_to_keep]
    if other_notes:
        other_sum = species_data[other_notes].sum()
        if other_sum > 0:
            grouped_data['Other'] = other_sum
    
    df_species_grouped_split[col] = grouped_data

df_species_grouped_split = df_species_grouped_split.fillna(0).astype('int32')
df_species_grouped_split = df_species_grouped_split.reindex(columns=split_species_order)

# Calculate percentages for subplot 1
df_pct_species_split = df_species_grouped_split.copy()
for col in df_pct_species_split.columns:
    species_total = df_pct_species_split[col].sum()
    if species_total > 0:
        df_pct_species_split[col] = (df_pct_species_split[col] / species_total * 100).round(2)

# Extend species_colors for the split keys
# Both populations of palmarum get the same color
species_colors_split = species_colors.copy()
species_colors_split['F. palmarum (India)'] = species_colors['F. palmarum']
species_colors_split['F. palmarum (Sri Lanka)'] = species_colors['F. palmarum']

# Define note orders for split keys (same as original palmarum)
species_orders_split = species_orders.copy()
species_orders_split['F. palmarum (India)'] = species_orders['F. palmarum']
species_orders_split['F. palmarum (Sri Lanka)'] = species_orders['F. palmarum']

# Create the figure
fig_split, axes_split = plt.subplots(nrows=1, ncols=2, figsize=(18, 8), dpi=100)

# Plot 1
colors_subplot0_split = [species_colors_split[spp] for spp in split_species_order if spp in df_pct_all_split.index or spp in df_pct_all_split.columns]
df_pct_all_split.plot.barh(ax=axes_split[0], color=colors_subplot0_split, stacked=True,
                     title='\nNote abundance across populations\n', rot=0)
axes_split[0].set_xlabel('% of note count across all recordings')

# Plot 2
x_positions_split = np.arange(len(df_pct_species_split.columns))
notes_in_legend_split = set()

for i, species_col in enumerate(df_pct_species_split.columns):
    order = species_orders_split.get(species_col, [])
    current_bottom = 0
    for note in order:
        if note in df_pct_species_split.index:
            value = df_pct_species_split.loc[note, species_col]
            if value > 0:
                axes_split[1].bar(i, value, bottom=current_bottom, 
                           color=note_color_map.get(note, 'gray'),
                           label=note if note not in notes_in_legend_split else '')
                notes_in_legend_split.add(note)
                current_bottom += value

axes_split[1].set_xticks(x_positions_split)
axes_split[1].set_xticklabels(df_pct_species_split.columns, rotation=45, ha='right')
axes_split[1].set_ylabel('% of notes for that species/population')
axes_split[1].set_title('\nNote distribution within populations\n')
axes_split[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Save Split Figure
png_path_split = os.path.join(output_dir, 'note_distribution_split.png')
plt.savefig(png_path_split, dpi=300, bbox_inches='tight')
print(f"Split note distribution saved as PNG to: {png_path_split}")

svg_path_split = os.path.join(output_dir, 'note_distribution_split.svg')
plt.savefig(svg_path_split, format='svg', bbox_inches='tight')
print(f"Split note distribution saved as SVG to: {svg_path_split}")

plt.close(fig_split)

