import pandas as pd
import numpy as np
import folium
from folium.plugins import FloatImage
from branca.colormap import LinearColormap

# Load prediction with weather
df = pd.read_csv("fire_prob_today.csv")
threshold = 0.2
timesteps = [1, 2, 3, 6, 12]
max_burn_duration = 3  # hours

# Set up grid
lats = np.sort(df['latitude'].unique())
lons = np.sort(df['longitude'].unique())
lat_to_idx = {lat: i for i, lat in enumerate(lats)}
lon_to_idx = {lon: i for i, lon in enumerate(lons)}
grid = np.zeros((len(lats), len(lons)))  # 0 = unburnt, 1 = burning
burn_time = np.zeros_like(grid)         # How long a cell has burned
wind_dir_grid = np.full_like(grid, np.nan)
humidity_grid = np.full_like(grid, np.nan)

# Initialize fire, wind, humidity
for _, row in df.iterrows():
    i, j = lat_to_idx[row["latitude"]], lon_to_idx[row["longitude"]]
    if row["fire_prob"] >= threshold:
        grid[i, j] = 1
        burn_time[i, j] = 1
    wind_dir_grid[i, j] = row.get("wind_direction", 0)
    humidity_grid[i, j] = (row.get("humidity_max", 0) + row.get("humidity_min", 0)) / 2

# Fire spread with wind and humidity
def spread_fire(grid, burn_time, wind_dir_grid, humidity_grid):
    new_grid = grid.copy()
    new_burn_time = burn_time.copy()
    rows, cols = grid.shape

    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 1:
                if burn_time[i, j] >= max_burn_duration:
                    continue  # burnt out
                new_burn_time[i, j] += 1
                wind_deg = wind_dir_grid[i, j]
                humidity = humidity_grid[i, j]

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            if grid[ni, nj] == 0:
                                # Compute spread angle vs wind
                                angle = np.arctan2(-di, dj) * 180 / np.pi
                                angle = (angle + 360) % 360
                                diff = abs(wind_deg - angle)
                                diff = min(diff, 360 - diff)

                                # Base chance
                                if diff <= 45:
                                    chance = 0.8
                                elif diff <= 90:
                                    chance = 0.5
                                else:
                                    chance = 0.2

                                # Humidity reduces chance
                                if humidity >= 70:
                                    chance *= 0.3
                                elif humidity >= 50:
                                    chance *= 0.6

                                if np.random.rand() < chance:
                                    new_grid[ni, nj] = 1
                                    new_burn_time[ni, nj] = 1
    return new_grid, new_burn_time

# Run simulation
history = [grid.copy()]
burn_tracker = [burn_time.copy()]
for step in range(1, max(timesteps) + 1):
    new_grid, new_burn_time = spread_fire(history[-1], burn_tracker[-1], wind_dir_grid, humidity_grid)
    history.append(new_grid)
    burn_tracker.append(new_burn_time)

# Base map
m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=10)
colormap = LinearColormap(['blue', 'lime', 'yellow', 'orange'], vmin=0, vmax=1)
colormap.caption = "🔥 Initial Fire Probability"
colormap.add_to(m)

# Original fire risk layer
base_layer = folium.FeatureGroup(name="Initial Fire Risk", show=True)
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=5,
        color=colormap(row["fire_prob"]),
        fill=True,
        fill_color=colormap(row["fire_prob"]),
        fill_opacity=0.7,
        popup=f"🔥 Risk: {row['fire_prob']:.2f}"
    ).add_to(base_layer)
base_layer.add_to(m)

# Spread layers
for t in timesteps:
    layer = folium.FeatureGroup(name=f"🔥 Spread after {t}h", show=False)
    grid_t = history[t]
    for i in range(grid_t.shape[0]):
        for j in range(grid_t.shape[1]):
            if grid_t[i, j] == 1:
                lat = lats[i]
                lon = lons[j]
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.9,
                    popup=f"🔥 Burnt at {t}h"
                ).add_to(layer)
    layer.add_to(m)

folium.LayerControl().add_to(m)
m.save("fire_spread_burnout_humidity.html")
print("✅ Saved: fire_spread_burnout_humidity.html")
