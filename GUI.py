import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load models and preprocessors
# RandomForest Model for Weather Category
weather_df = pd.read_csv(r"C:\Users\Hello\DS Project\Cleaned_dataset\weather_data_with_hour.csv")
X = weather_df.drop(['weather_code', 'weather_category', 'date_only', 'hour'], axis='columns')
y = weather_df['weather_category']

sc_weather = StandardScaler()
X_pp = sc_weather.fit_transform(X)

le_weather = LabelEncoder()
y_pp = le_weather.fit_transform(y)

rfc_model = RandomForestClassifier(
    criterion='gini',
    min_samples_split=10,
    n_estimators=49,
    min_samples_leaf=2,
    max_depth=10
)
rfc_model.fit(X_pp, y_pp)

# Temperature Forecast Model
temperature_model = load_model(r"C:\Users\Hello\DS Project\temperature_forecast_model.keras")
with open(r"temperature_scaler.pkl", 'rb') as file:
    temperature_scaler = pickle.load(file)

df = pd.read_csv(r"C:\Users\Hello\DS Project\Cleaned_dataset\weather_data_with_hour.csv")
df['datetime'] = pd.to_datetime(df['date_only'].astype(str) + ' ' + df['hour'].astype(str) + ':00')

# Helper Functions
def prepare_sequence(last_rows, lookback):
    sequence = np.array(last_rows).reshape(-1, 1)
    scaled_sequence = temperature_scaler.transform(sequence)
    return scaled_sequence.reshape(1, lookback, 1)

def get_last_24_hours(current_time):
    last_hour_time = current_time.replace(minute=0, second=0, microsecond=0)
    start_time = last_hour_time - timedelta(hours=24)
    df_filtered = df[(df['datetime'] >= start_time) & (df['datetime'] < last_hour_time)]

    if df_filtered.empty or len(df_filtered) < 24:
        print("Fetching data from last year...")
        last_year_time = last_hour_time.replace(year=last_hour_time.year - 1)
        start_time_last_year = last_year_time - timedelta(hours=24)
        df_filtered = df[(df['datetime'] >= start_time_last_year) & (df['datetime'] < last_year_time)]

    if len(df_filtered) < 24:
        print("Insufficient data. Filling gaps with average temperature...")
        missing_rows = 24 - len(df_filtered)
        avg_temp = df['temperature_2m'].mean()
        dummy_data = np.full((missing_rows, 1), avg_temp)
        last_24_hours = np.vstack((dummy_data, df_filtered[['temperature_2m']].values))[:24]
    else:
        last_24_hours = df_filtered[['temperature_2m']].values[-24:]

    return last_24_hours

def predict_weather_category():
    try:
        temp = float(temp_entry.get())
        humidity = float(humidity_entry.get())
        wind_speed = float(wind_speed_entry.get())
        rain = float(rain_entry.get())

        input_data = np.array([[temp, humidity, wind_speed, rain]])
        input_scaled = sc_weather.transform(input_data)
        prediction = rfc_model.predict(input_scaled)
        category = le_weather.inverse_transform(prediction)[0]

        messagebox.showinfo("Prediction Result", f"Predicted Weather Category: {category}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")

def forecast_temperature():
    current_time = datetime.now()
    next_full_hour = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    last_24_hours = get_last_24_hours(next_full_hour)

    if last_24_hours is None:
        print("Error: Could not retrieve 24-hour data.")
        return

    forecast_results = []
    remaining_hours = [next_full_hour + timedelta(hours=i) for i in range(0, 24 - next_full_hour.hour)]

    for time_point in remaining_hours:
        sequence = prepare_sequence(last_24_hours, lookback=24)
        temp_prediction = temperature_model.predict(sequence)
        temp_rescaled = temperature_scaler.inverse_transform(temp_prediction.reshape(-1, 1))
        temp_rescaled_rounded = round(temp_rescaled[0, 0], 1)
        forecast_results.append((time_point.strftime("%H:%M"), temp_rescaled_rounded))
        last_24_hours = np.append(last_24_hours, [[temp_rescaled_rounded]], axis=0)[1:]

    hourly_output.delete(*hourly_output.get_children())
    for entry in forecast_results:
        hourly_output.insert("", "end", values=entry)

# GUI Setup
root = tk.Tk()
root.title("Integrated Weather Application")
root.geometry("900x700")

# Weather Category Prediction
category_frame = tk.Frame(root, padx=10, pady=10)
category_frame.pack(fill="x", pady=10)

category_label = tk.Label(category_frame, text="Weather Category Predictor", font=("Helvetica", 16, "bold"))
category_label.pack()

input_frame = tk.Frame(category_frame)
input_frame.pack(pady=10)

temp_label = tk.Label(input_frame, text="Temperature (°C):")
temp_label.grid(row=0, column=0, padx=5, pady=5)
temp_entry = tk.Entry(input_frame)
temp_entry.grid(row=0, column=1, padx=5, pady=5)

humidity_label = tk.Label(input_frame, text="Humidity (%):")
humidity_label.grid(row=1, column=0, padx=5, pady=5)
humidity_entry = tk.Entry(input_frame)
humidity_entry.grid(row=1, column=1, padx=5, pady=5)

wind_speed_label = tk.Label(input_frame, text="Wind Speed (m/s):")
wind_speed_label.grid(row=2, column=0, padx=5, pady=5)
wind_speed_entry = tk.Entry(input_frame)
wind_speed_entry.grid(row=2, column=1, padx=5, pady=5)

rain_label = tk.Label(input_frame, text="Rain (mm):")
rain_label.grid(row=3, column=0, padx=5, pady=5)
rain_entry = tk.Entry(input_frame)
rain_entry.grid(row=3, column=1, padx=5, pady=5)

predict_button = tk.Button(category_frame, text="Predict Weather Category", command=predict_weather_category)
predict_button.pack(pady=5)

# Temperature Forecast
forecast_frame = tk.Frame(root, padx=10, pady=10)
forecast_frame.pack(fill="x", pady=10)

temperature_label = tk.Label(forecast_frame, text="Temperature Forecast", font=("Helvetica", 16, "bold"))
temperature_label.pack()

hourly_label = tk.Label(forecast_frame, text="Hourly Forecast (Remaining Hours Today):")
hourly_label.pack(pady=5)

hourly_output = ttk.Treeview(forecast_frame, columns=("Time", "Temperature (°C)"), show="headings")
hourly_output.heading("Time", text="Time")
hourly_output.heading("Temperature (°C)", text="Temperature (°C)")
hourly_output.pack(pady=5, fill="x")

forecast_button = tk.Button(forecast_frame, text="Generate Hourly Forecast", command=forecast_temperature)
forecast_button.pack(pady=5)

root.mainloop()