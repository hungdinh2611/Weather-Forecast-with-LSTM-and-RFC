# Weather-Forecast-with-LSTM-and-RFC

This repository contains a Python-based GUI application for weather prediction and forecasting. It leverages machine learning models and LSTM to predict weather categories and forecast temperatures, providing both hourly and daily forecasts for up to 7 days.

## Features

### 1. Predict Weather Category
- Predicts the weather category (e.g., sunny, rainy) based on user inputs such as temperature, humidity, wind speed, and rainfall.
- Utilizes a Random Forest Classifier trained on historical weather data.

### 2. Hourly Temperature Forecast
- Predicts the temperature for the remaining hours of the current day.
- Uses a trained neural network model for temperature forecasting.

### 3. 7-Day Temperature Forecast
- Provides the minimum and maximum temperatures for the next 7 days.
- Forecasts daily temperature trends using a neural network.

## Installation

### Prerequisites
- Python 3.8+
- Required Python libraries: `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `tkinter`
- Trained models and scalers:
  - `temperature_forecast_model.keras`
  - `temperature_scaler.pkl`
  - `weather_data_with_hour.csv`

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/weather-forecast.git
   cd weather-forecast
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the required model and data files in the specified paths:
   - `temperature_forecast_model.keras`
   - `temperature_scaler.pkl`
   - `weather_data_with_hour.csv`

## Usage

1. Run the application:
   ```bash
   python app.py
   ```

2. Input data in the GUI fields for weather category prediction.

3. Generate hourly forecasts for the current day.

4. Generate a 7-day forecast to view minimum and maximum temperatures for the next week.

## File Structure
```
weather-forecast/
├── app.py                      # Main application script
├── requirements.txt            # Dependencies
├── temperature_forecast_model.keras  # Temperature forecasting model
├── temperature_scaler.pkl      # Scaler for temperature data
├── weather_data_with_hour.csv  # Historical weather dataset
└── README.md                   # Project documentation
```

## Models
### Random Forest Classifier
- Trained to classify weather categories based on features like temperature, humidity, wind speed, and rainfall.

### Neural Network for Temperature Forecasting
- Trained to predict hourly temperatures based on historical data using a sequence-to-sequence approach.

## Future Improvements
- Add support for more weather features such as pressure and UV index.
- Enhance the user interface for a more intuitive experience.
- Integrate real-time weather API for dynamic data updates.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- [Scikit-learn](https://scikit-learn.org/)
- [TensorFlow](https://www.tensorflow.org/)
- Weather data provided by [your data source].
