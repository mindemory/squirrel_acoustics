import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from params import PROJECT_PATH

# Load Location File
location_path = os.path.join(PROJECT_PATH, 'locations_dec.csv')
location_df = pd.read_csv(location_path)

# Add extension to file names wherever needed
for i in range(location_df.shape[0]):
    if location_df.loc[i, '12_Audio_file_name'][-4:] != '.wav' and location_df.loc[i, '12_Audio_file_name'][-4:] != '.WAV':
        location_df.loc[i, '12_Audio_file_name'] = location_df.loc[i, '12_Audio_file_name'] + '.wav'

# Load file_df which has Species, Location, Latitude, Longitude information
df_folder = os.path.join(PROJECT_PATH, 'dataframes')
file_df = pd.read_csv(os.path.join(df_folder, 'file_df.csv'))

# Load high-resolution Natural Earth data (50m resolution)
# Check if we have it cached locally first
ne_cache_path = os.path.join(PROJECT_PATH, 'Figures', 'Geoanalysis', 'ne_50m_admin_0_countries.shp')

if os.path.exists(ne_cache_path):
    print("Loading high-resolution Natural Earth data from cache...")
    world = gpd.read_file(ne_cache_path)
else:
    # Download 50m resolution Natural Earth data
    try:
        import urllib.request
        import zipfile
        import tempfile
        
        print("Downloading high-resolution (50m) Natural Earth data...")
        ne_url = 'https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip'
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, 'ne_50m_countries.zip')
        
        try:
            import requests
            response = requests.get(ne_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=60, stream=True)
            if response.status_code == 200:
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Extract the zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find the shapefile
                shp_path = os.path.join(temp_dir, 'ne_50m_admin_0_countries.shp')
                if os.path.exists(shp_path):
                    world = gpd.read_file(shp_path)
                    # Cache it locally for future use
                    cache_dir = os.path.dirname(ne_cache_path)
                    os.makedirs(cache_dir, exist_ok=True)
                    # Copy all shapefile components
                    import shutil
                    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                        src = shp_path.replace('.shp', ext)
                        dst = ne_cache_path.replace('.shp', ext)
                        if os.path.exists(src):
                            shutil.copy2(src, dst)
                    print("High-resolution data downloaded and cached successfully")
                else:
                    raise Exception("Shapefile not found in downloaded archive")
            else:
                raise Exception(f"Download failed with status code {response.status_code}")
        except ImportError:
            # Fallback to urllib
            urllib.request.urlretrieve(ne_url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            shp_path = os.path.join(temp_dir, 'ne_50m_admin_0_countries.shp')
            if os.path.exists(shp_path):
                world = gpd.read_file(shp_path)
                # Cache it locally
                cache_dir = os.path.dirname(ne_cache_path)
                os.makedirs(cache_dir, exist_ok=True)
                import shutil
                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                    src = shp_path.replace('.shp', ext)
                    dst = ne_cache_path.replace('.shp', ext)
                    if os.path.exists(src):
                        shutil.copy2(src, dst)
                print("High-resolution data downloaded and cached successfully")
            else:
                raise Exception("Shapefile not found in downloaded archive")
    
    except Exception as e:
        print(f"Could not download high-resolution data: {e}")
        print("Falling back to low-resolution dataset...")
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

# Filter for India and Sri Lanka
countries = ["India", "Sri Lanka"]
map_df = world[world["NAME"].isin(countries)]

# Load Indian state boundaries (admin level 1)
# Check if we have it cached locally first
states_cache_path = os.path.join(PROJECT_PATH, 'Figures', 'Geoanalysis', 'ne_50m_admin_1_states_provinces.shp')

if os.path.exists(states_cache_path):
    print("Loading Indian state boundaries from cache...")
    states_world = gpd.read_file(states_cache_path)
else:
    # Download 50m resolution Natural Earth admin level 1 data
    try:
        import urllib.request
        import zipfile
        import tempfile
        
        print("Downloading state boundaries (50m) Natural Earth data...")
        ne_states_url = 'https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_1_states_provinces.zip'
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, 'ne_50m_states.zip')
        
        try:
            import requests
            response = requests.get(ne_states_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=60, stream=True)
            if response.status_code == 200:
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Extract the zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find the shapefile
                shp_path = os.path.join(temp_dir, 'ne_50m_admin_1_states_provinces.shp')
                if os.path.exists(shp_path):
                    states_world = gpd.read_file(shp_path)
                    # Cache it locally for future use
                    cache_dir = os.path.dirname(states_cache_path)
                    os.makedirs(cache_dir, exist_ok=True)
                    # Copy all shapefile components
                    import shutil
                    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                        src = shp_path.replace('.shp', ext)
                        dst = states_cache_path.replace('.shp', ext)
                        if os.path.exists(src):
                            shutil.copy2(src, dst)
                    print("State boundaries downloaded and cached successfully")
                else:
                    raise Exception("Shapefile not found in downloaded archive")
            else:
                raise Exception(f"Download failed with status code {response.status_code}")
        except ImportError:
            # Fallback to urllib
            urllib.request.urlretrieve(ne_states_url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            shp_path = os.path.join(temp_dir, 'ne_50m_admin_1_states_provinces.shp')
            if os.path.exists(shp_path):
                states_world = gpd.read_file(shp_path)
                # Cache it locally
                cache_dir = os.path.dirname(states_cache_path)
                os.makedirs(cache_dir, exist_ok=True)
                import shutil
                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                    src = shp_path.replace('.shp', ext)
                    dst = states_cache_path.replace('.shp', ext)
                    if os.path.exists(src):
                        shutil.copy2(src, dst)
                print("State boundaries downloaded and cached successfully")
            else:
                raise Exception("Shapefile not found in downloaded archive")
    
    except Exception as e:
        print(f"Could not download state boundaries: {e}")
        states_world = None

# Filter for Indian states
if 'states_world' in locals() and states_world is not None:
    # Filter for India using the 'admin' column
    if 'admin' in states_world.columns:
        india_states = states_world[states_world['admin'] == 'India'].copy()
        print(f"Found {len(india_states)} Indian states/provinces")
    else:
        print("Warning: 'admin' column not found in states data")
        print("Available columns:", states_world.columns.tolist()[:10])
        india_states = None
else:
    india_states = None

# Get unique locations with species and count recordings per location per species
location_species_counts = file_df.groupby(['Location', 'Species', 'Latitude', 'Longitude']).size().reset_index(name='num_recordings')

# Define species colors
species_colors = {
    'F. palmarum': '#98003F',
    'F. pennanti': 'orange',
    'F. tristriatus': '#5D4EA0',
    'F. sublineatus': '#55ab0f'
}

# Create the map
fig, ax = plt.subplots(figsize=(10, 10))

# Plot India and Sri Lanka map
map_df.plot(ax=ax, color='white', edgecolor='black', linewidth=1)

# Overlay Indian state boundaries if available
if india_states is not None and len(india_states) > 0:
    india_states.plot(ax=ax, color='none', edgecolor='#333333', linewidth=0.5, alpha=0.6)

# Load and plot IUCN species distribution layers
iucn_layers_path = os.path.join(os.path.dirname(PROJECT_PATH), 'iucn_layers')
species_dir_mapping = {
    'palmarum': 'F. palmarum',
    'tristriatus': 'F. tristriatus',
    'pennanti': 'F. pennanti',
    'sublineatus': 'F. sublineatus'
}

# Plot distribution layers in the desired order
for dir_name in ['palmarum', 'tristriatus', 'pennanti', 'sublineatus']:
    species_name = species_dir_mapping[dir_name]
    shapefile_path = os.path.join(iucn_layers_path, dir_name, 'data_0.shp')
    
    if os.path.exists(shapefile_path):
        try:
            distribution_gdf = gpd.read_file(shapefile_path)
            color = species_colors.get(species_name, 'gray')
            # Plot with same color as scatter points, semi-transparent
            distribution_gdf.plot(ax=ax, color=color, alpha=0.3, edgecolor=color, linewidth=0.5)
        except Exception as e:
            print(f"Warning: Could not load distribution layer for {species_name}: {e}")

# Plot locations with size directly proportional to number of recordings
# Size = num_recordings * 5 (matplotlib s parameter is area in points^2)
# For radius = num_recordings * 5, we use area = π * r^2, but for simplicity
# we'll use s = num_recordings * 5 * scale_factor
# Using a scale factor to make sizes reasonable (since s is in points^2)
scale_factor = 10  # Adjust this to make points appropriately sized
base_size = 5  # Base multiplier

# Define the desired species order
desired_order = ['F. palmarum', 'F. tristriatus', 'F. pennanti', 'F. sublineatus']

# Get available species in the desired order
available_species = [s for s in desired_order if s in location_species_counts['Species'].unique()]

# Plot each species separately with its color in the specified order
for species in available_species:
    species_data = location_species_counts[location_species_counts['Species'] == species]
    # Size = num_recordings * 5 * scale_factor
    species_sizes = species_data['num_recordings'] * base_size * scale_factor
    color = species_colors.get(species, 'gray')
    ax.scatter(species_data['Longitude'], 
               species_data['Latitude'], 
               c=color, s=species_sizes, alpha=0.7, label=species)

# Create custom legend with fixed size markers (size = 5) in the specified order
from matplotlib.lines import Line2D
legend_elements = []
for species in available_species:
    color = species_colors.get(species, 'gray')
    # Fixed size for legend: base_size (which is 5)
    # markersize is in points, so we use a reasonable size
    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=color, markersize=8,
                                   label=species, alpha=0.7))

# Add legend at top-right with custom handles
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
          title='Species', title_fontsize=12)

# Set title
ax.set_title('Recording Locations Map', fontsize=14, fontweight='bold')

# Remove axes
ax.axis('off')

# Save the figure in both PNG and SVG formats
output_dir = os.path.join(PROJECT_PATH, 'Figures')
os.makedirs(output_dir, exist_ok=True)

# Save as PNG
png_path = os.path.join(output_dir, 'recording_locations_map.png')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
print(f"Map saved as PNG to: {png_path}")

# Save as SVG (editable format)
svg_path = os.path.join(output_dir, 'recording_locations_map.svg')
plt.savefig(svg_path, format='svg', bbox_inches='tight')
print(f"Map saved as SVG to: {svg_path}")

plt.show()

