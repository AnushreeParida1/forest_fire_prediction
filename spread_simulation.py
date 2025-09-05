import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

# Load predicted fire probabilities (from heat_map.py result)
df = pd.read_csv("fire_prob_today.csv")  # <-- Save this in heat_map.py next time!

# Parameters
grid_size = 30  # meters
threshold = 0.8  # cells with prob > 0.8 are ignition sources
timesteps = [1, 2, 3, 6, 12]

# Step 1: Convert lat/lon to 2D grid (index based)
lats = np.sort(df['latitude'].unique())
lons = np.sort(df['longitude'].unique())

lat_to_idx = {lat: i for i, lat in enumerate(lats)}
lon_to_idx = {lon: i for i, lon in enumerate(lons)}

grid = np.zeros((len(lats), len(lons)))

# Step 2: Initialize fire from high-risk cells
for _, row in df.iterrows():
    if row["fire_prob"] > threshold:
        i = lat_to_idx[row["latitude"]]
        j = lon_to_idx[row["longitude"]]
        grid[i, j] = 1  # Fire starts here

# Helper to spread fire to neighbors
def spread_fire(grid):
    new_grid = grid.copy()
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 1:
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            if grid[ni, nj] == 0:
                                # Simple random chance to catch fire
                                if np.random.rand() < 0.4:
                                    new_grid[ni, nj] = 1
    return new_grid

# Simulate and store results
history = [grid.copy()]
for step in range(1, max(timesteps)+1):
    new_grid = spread_fire(history[-1])
    history.append(new_grid)

# Plot snapshots
for hr in timesteps:
    plt.figure(figsize=(6, 6))
    sns.heatmap(history[hr], cmap="OrRd", cbar=False)
    plt.title(f"🔥 Fire Spread after {hr} hour(s)")
    plt.axis("off")
    plt.savefig(f"spread_{hr}h.png")
    plt.close()

# Optional: Create animation
fig, ax = plt.subplots()
heatmap = ax.imshow(history[0], cmap='OrRd')

def update(frame):
    heatmap.set_data(history[frame])
    ax.set_title(f"🔥 Fire Spread - Hour {frame}")
    return [heatmap]

anim = FuncAnimation(fig, update, frames=len(history), interval=1000)
anim.save("fire_spread_simulation.gif", writer='pillow')
print("✅ Fire spread simulation saved as fire_spread_simulation.gif")
