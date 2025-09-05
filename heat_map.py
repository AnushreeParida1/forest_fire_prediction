import pandas as pd
import requests
import folium
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime
from branca.colormap import LinearColormap
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1️⃣ Load trained model and scaler
model = load_model("pauri_fire_model_final.h5")
scaler = joblib.load("pauri_scaler.pkl")

# 2️⃣ Select prediction date (change this to desired date)
input_date = "2025-09-05"
date_obj = datetime.strptime(input_date, "%Y-%m-%d")
day, month, dayofweek = date_obj.day, date_obj.month, date_obj.weekday()

# 3️⃣ Load grid coordinates
grid_df = pd.read_csv("pauri_grid.csv")

# 4️⃣ Define weather fetching function
def fetch_weather(row):
    lat, lon = row["latitude"], row["longitude"]
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&start_date={input_date}&end_date={input_date}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
        "relative_humidity_2m_max,relative_humidity_2m_min,windspeed_10m_max,"
        "winddirection_10m_dominant"

    )
    try:
        r = requests.get(url, timeout=10)
        data = r.json()["daily"]
        return {
            "latitude": lat,
            "longitude": lon,
            "temperature_max": data["temperature_2m_max"][0],
            "temperature_min": data["temperature_2m_min"][0],
            "humidity_max": data["relative_humidity_2m_max"][0],
            "humidity_min": data["relative_humidity_2m_min"][0],
            "precipitation_sum": data["precipitation_sum"][0],
            "windspeed_max": data["windspeed_10m_max"][0],
            "wind_direction": data["winddirection_10m_dominant"][0],

        }
    except Exception as e:
        print(f"❌ Weather fetch error for {lat}, {lon}: {e}")
        return None

# 5️⃣ Fetch weather in parallel
print("🌦️ Fetching weather data for grid points...")
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(fetch_weather, row) for _, row in grid_df.iterrows()]
    weather_data = [f.result() for f in as_completed(futures) if f.result()]

weather_df = pd.DataFrame(weather_data)
if weather_df.empty:
    raise Exception("❌ No weather data fetched.")

# 6️⃣ Prepare features for prediction
df = pd.merge(grid_df, weather_df, on=["latitude", "longitude"])
df["day"] = day
df["month"] = month
df["dayofweek"] = dayofweek

# Add dummy MODIS features
for col in ['brightness', 'bright_t31', 'frp']:
    df[col] = 0

features = [
    'brightness', 'bright_t31', 'frp', 'day', 'month', 'dayofweek',
    'temperature_max', 'temperature_min', 'humidity_max', 'humidity_min',
    'precipitation_sum', 'windspeed_max'
]

X = df[features]
X_scaled = scaler.transform(X)

# 7️⃣ Predict fire probabilities
df["fire_prob"] = model.predict(X_scaled).flatten()

# 🔁 Save prediction to CSV for use in spread simulation
df.to_csv("fire_prob_today.csv", index=False)
print("💾 Saved fire probabilities to fire_prob_today.csv")

# 8️⃣ Generate heatmap
print("🗺️ Generating heatmap...")
m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=10)
colormap = LinearColormap(
    ['blue', 'lime', 'yellow', 'orange', 'red', 'darkred'],
    index=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    vmin=0.0, vmax=1.0
).to_step(6)
colormap.caption = "🔥 Fire Risk Probability"
colormap.add_to(m)

for _, row in df.iterrows():
    prob = float(row["fire_prob"])
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=5,
        color=colormap(prob),
        fill=True,
        fill_color=colormap(prob),
        fill_opacity=0.8,
        popup=f"🔥 Risk: {prob:.2f}"
    ).add_to(m)

# 9️⃣ Save final output map
output_file = f"fire_live_map_{input_date}.html"
m.save(output_file)
print(f"✅ Fire heatmap saved: {output_file}")
