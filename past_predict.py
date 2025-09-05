import pandas as pd
import requests
import folium
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime
from branca.colormap import LinearColormap
import time

# 1️⃣ Load model and scaler
model = load_model("pauri_fire_model_final.h5")
scaler = joblib.load("pauri_scaler.pkl")

# 2️⃣ Set date (today or future date)
input_date = "2021-04-13"  # Change this as needed
date_obj = datetime.strptime(input_date, "%Y-%m-%d")
day, month, dayofweek = date_obj.day, date_obj.month, date_obj.weekday()

# 3️⃣ Load grid points
grid_df = pd.read_csv("pauri_grid.csv")  # Must contain 'latitude' and 'longitude'

# 4️⃣ Fetch weather for each grid
def fetch_weather(lat, lon, date):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&start_date={date}&end_date={date}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
        "relative_humidity_2m_max,relative_humidity_2m_min,windspeed_10m_max"
        "&timezone=auto"
    )
    try:
        r = requests.get(url)
        data = r.json()["daily"]
        return {
            "temperature_max": data["temperature_2m_max"][0],
            "temperature_min": data["temperature_2m_min"][0],
            "humidity_max": data["relative_humidity_2m_max"][0],
            "humidity_min": data["relative_humidity_2m_min"][0],
            "precipitation_sum": data["precipitation_sum"][0],
            "windspeed_max": data["windspeed_10m_max"][0],
        }
    except:
        return None

# 5️⃣ Apply weather fetching to each row
weather_data = []
print("Fetching weather data for each grid point...")
for _, row in grid_df.iterrows():
    w = fetch_weather(row["latitude"], row["longitude"], input_date)
    if w:
        weather_data.append({**w, "latitude": row["latitude"], "longitude": row["longitude"]})
    else:
        print(f"⚠️ Failed to fetch data for lat={row['latitude']}, lon={row['longitude']}")
    time.sleep(0.5)  # prevent spamming API

weather_df = pd.DataFrame(weather_data)

# 6️⃣ Merge weather into grid dataframe
df = pd.merge(grid_df, weather_df, on=["latitude", "longitude"], how="inner")
df["day"] = day
df["month"] = month
df["dayofweek"] = dayofweek

# 7️⃣ Add dummy MODIS values
for col in ['brightness', 'bright_t31', 'frp']:
    df[col] = 0

# 8️⃣ Scale and predict
features = [
    'brightness', 'bright_t31', 'frp', 'day', 'month', 'dayofweek',
    'temperature_max', 'temperature_min', 'humidity_max', 'humidity_min',
    'precipitation_sum', 'windspeed_max'
]
X = df[features]
X_scaled = scaler.transform(X)
df["fire_prob"] = model.predict(X_scaled).flatten()

# 9️⃣ Create heatmap
m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=10)

colormap = LinearColormap(
    colors=['blue', 'lime', 'yellow', 'orange', 'red', 'darkred'],
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
        popup=f"{prob:.2f}"
    ).add_to(m)

# 🔟 Save output
output_file = f"fire_live_map_{input_date}.html"
m.save(output_file)
print(f"✅ Saved live fire risk map as {output_file}")
