import pandas as pd
import numpy as np
import requests
import folium
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime
from branca.colormap import LinearColormap

# 1️⃣ Load model and scaler
model = load_model("pauri_fire_model_final.h5")
scaler = joblib.load("pauri_scaler.pkl")

# 2️⃣ Choose prediction date
input_date = "2025-07-06"  # 🔁 Change to any date in YYYY-MM-DD format
date_obj = datetime.strptime(input_date, "%Y-%m-%d")
day, month, dayofweek = date_obj.day, date_obj.month, date_obj.weekday()

# 3️⃣ Load coordinate grid
grid_df = pd.read_csv("pauri_grid.csv")  # Should contain 'latitude' and 'longitude'

# 4️⃣ Fetch weather data for the date
sample_lat = grid_df['latitude'].mean()
sample_lon = grid_df['longitude'].mean()

weather_url = (
    f"https://api.open-meteo.com/v1/forecast?"
    f"latitude={sample_lat}&longitude={sample_lon}&start_date={input_date}&end_date={input_date}"
    "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_max,"
    "relative_humidity_2m_min,windspeed_10m_max&timezone=auto"
)

response = requests.get(weather_url)
data = response.json()["daily"]

weather_features = {
    "temperature_max": data["temperature_2m_max"][0],
    "temperature_min": data["temperature_2m_min"][0],
    "humidity_max": data["relative_humidity_2m_max"][0],
    "humidity_min": data["relative_humidity_2m_min"][0],
    "precipitation_sum": data["precipitation_sum"][0],
    "windspeed_max": data["windspeed_10m_max"][0],
}

# 5️⃣ Add features to each grid point
for col, val in weather_features.items():
    grid_df[col] = val
grid_df["day"] = day
grid_df["month"] = month
grid_df["dayofweek"] = dayofweek

# 6️⃣ Add MODIS placeholders (optional, if not present)
for col in ['brightness', 'bright_t31', 'frp']:
    if col not in grid_df.columns:
        grid_df[col] = 0

# 7️⃣ Prepare data for prediction
features = [
    'brightness', 'bright_t31', 'frp', 'day', 'month', 'dayofweek',
    'temperature_max', 'temperature_min', 'humidity_max', 'humidity_min',
    'precipitation_sum', 'windspeed_max'
]

X_input = grid_df[features]
X_scaled = scaler.transform(X_input)

# 8️⃣ Predict fire probabilities
y_probs = model.predict(X_scaled).flatten()
grid_df["fire_prob"] = y_probs

# 9️⃣ Build folium heatmap with custom color scale
center = [grid_df['latitude'].mean(), grid_df['longitude'].mean()]
m = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")

# 🔥 Custom color gradient: Blue → Lime → Yellow → Orange → Red → DarkRed
color_steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
colors = ['blue', 'lime', 'yellow', 'orange', 'red', 'darkred']

# ✅ Sort and convert to lists
sorted_pairs = sorted(zip(color_steps, colors))
# color_steps, colors = map(list, zip(*sorted_pairs))
color_steps, colors = map(list, zip(*sorted_pairs))
color_steps = [float(c) for c in color_steps]  

# ✅ Build colormap
colormap = LinearColormap(
    colors=colors,
    index=color_steps,
    vmin=0.0,
    vmax=1.0
).to_step(n=len(color_steps))  # 👈 converts it to a safe stepped map

colormap.caption = "🔥 Fire Risk Probability"
colormap.add_to(m)

# 🔵🔴 Plot points
for _, row in grid_df.iterrows():
    prob = float(row["fire_prob"])
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=5,
        color=colormap(prob),
        fill=True,
        fill_color=colormap(prob),
        fill_opacity=0.8,
        popup=f"🔥 {prob:.2f}"
    ).add_to(m)

# 🔟 Save output
output_file = f"fire_map_{input_date}.html"
m.save(output_file)
print(f"✅ Fire heatmap saved as: {output_file}")

