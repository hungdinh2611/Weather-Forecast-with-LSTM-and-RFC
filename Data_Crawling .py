import requests
import pandas as pd
import time
from datetime import datetime, timedelta

# Define a simple cache mechanism (for example, using a dictionary)
cache = {}

# Function to make requests with retries
def fetch_data_with_retries(url, params, retries=5, backoff_factor=0.2):
    attempt = 0
    while attempt < retries:
        try:
            # Check cache first
            cache_key = f"{url}?{requests.models.PreparedRequest().prepare_url(url, params)}"
            if cache_key in cache:
                return cache[cache_key]

            # If not cached, make the request
            response = requests.get(url, params=params)
            response.raise_for_status()  # Check for request errors (e.g., 404 or 500)
            
            # Store the response in cache
            cache[cache_key] = response.json()
            return cache[cache_key]

        except (requests.RequestException, ValueError) as e:
            print(f"Error fetching data: {e}")
            attempt += 1
            if attempt < retries:
                time.sleep(backoff_factor * (2 ** attempt))  # Exponential backoff
            else:
                raise

# Set up the parameters for the API request
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 21.0245,
    "longitude": 105.8412,
    "start_date": "2020-01-01",
    "end_date": "2024-09-30",
    "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_10m","weather_code"]
}

# Fetch the weather data with retry and caching
response_data = fetch_data_with_retries(url, params)

# Print the response data to understand its structure
#(response_data)

# Assuming the response data has a 'hourly' key, you might need to adjust
if "hourly" in response_data:
    response = response_data  # No need to use index if it's a direct dict
    # Output location and timezone information
    print(f"Coordinates {response['latitude']}°N {response['longitude']}°E")
    print(f"Elevation {response['elevation']} m asl")
    print(f"Timezone {response['timezone']} {response['timezone_abbreviation']}")
    print(f"Timezone difference to GMT+0 {response['utc_offset_seconds']} s")

    # Process hourly data
    hourly_data = {
        "date": pd.to_datetime(response['hourly']['time'], utc=True),  # No need for unit="s"
        "temperature_2m": response['hourly']['temperature_2m'],
        "relative_humidity_2m": response['hourly']['relative_humidity_2m'],
        "rain": response['hourly']['rain'],
        "wind_speed_10m": response['hourly']['wind_speed_10m'],
        "weather_code":response['hourly']['weather_code']
    }

    # Convert the data into a DataFrame
    hourly_dataframe = pd.DataFrame(data=hourly_data)

    # Save the DataFrame to a CSV file
    hourly_dataframe.to_csv("weather_data.csv", index=False)

    print("Data saved to weather_data.csv")
else:
    print("Error: 'hourly' data not found in response.")
