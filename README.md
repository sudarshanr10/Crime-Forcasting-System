# Crime-Forcasting-System

# Crime Hotspot Forecasting Project

This repository contains Python code for building and analyzing a crime hotspot forecasting model using historical crime, demographic, and environmental data. The project supports both spatial clustering and real-time crime forecasting using the **DBSCAN** and **ARIMA** models.

---

## **Features**

### **Data Integration and Preprocessing**
- Combines multiple datasets (crime data, demographic data, weather data).
- Cleans data by removing unnecessary columns, handling missing values, and standardizing location and timestamp formats.
- Integrates structured data using SQL and unstructured data using MongoDB.

### **Spatial-Temporal Analysis**
- **Spatial Clustering**: Identifies geographic crime hotspots using the **DBSCAN** algorithm.
- **Temporal Forecasting**: Uses the **ARIMA** model to predict the likelihood of crimes in each hotspot over time.

### **Predictive Modeling**
- Forecasts the number of crimes per day for the next 30 days for each hotspot.
- Provides real-time capability for dynamic updates and enhanced crime predictions.

### **Evaluation Metrics**
- **Hotspot Detection**: Precision and recall to evaluate clustering accuracy.
- **Forecasting Accuracy**: Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) for ARIMA predictions.

---

## **Requirements**

The following Python libraries are used:

- `pandas`
- `numpy`
- `folium`
- `sqlite3`
- `scikit-learn`
- `statsmodels`
- `pmdarima`
- `psycopg2`

---

## **Setup Instructions**

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   ```

2. **Navigate to the project folder**:
   ```bash
   cd crime_hotspot_forecast
   ```

3. **Install required dependencies**:
   ```bash
   pip install pandas numpy folium scikit-learn statsmodels pmdarima psycopg2
   ```

---

## **Usage**

### **1. Data Preprocessing**
- Load and clean crime data from a CSV file:
  ```python
  import pandas as pd
  file_path = 'crimedatadenver.csv'
  crime_data = pd.read_csv(file_path)
  ```
- Remove unnecessary columns and save the cleaned data:
  ```python
  columns_to_remove = ['OBJECTID', 'INCIDENT_ID', 'OFFENSE_ID', 'OFFENSE_CODE']
  crime_data_cleaned = crime_data.drop(columns=columns_to_remove)
  crime_data_cleaned.to_csv("cleaned_crime_data.csv", index=False)
  ```

### **2. Database Integration**
- Insert cleaned data into an SQLite database:
  ```python
  import sqlite3
  conn = sqlite3.connect('crime_data.db')
  # Create and populate the table
  ```

### **3. Spatial Clustering with DBSCAN**
- Identify crime hotspots based on latitude and longitude:
  ```python
  from sklearn.cluster import DBSCAN
  dbscan = DBSCAN(eps=0.01, min_samples=5)
  crime_data_cleaned['cluster'] = dbscan.fit_predict(crime_data_cleaned[['latitude', 'longitude']])
  ```

### **4. Forecasting with ARIMA**
- Predict daily crime counts for the next 30 days:
  ```python
  from statsmodels.tsa.arima.model import ARIMA
  model = ARIMA(time_series, order=(1, 1, 1))
  model_fit = model.fit()
  forecast = model_fit.forecast(steps=30)
  ```

---

## **Key Functions**

- **`preprocess_data(data)`**: Cleans and prepares crime data for analysis.
- **`run_dbscan_clustering(data)`**: Applies DBSCAN to identify crime hotspots.
- **`run_arima_forecasting(data)`**: Forecasts crime counts using the ARIMA model.
- **`visualize_hotspots(data)`**: Visualizes crime clusters on an interactive map using Folium.

---

## **Example Workflow**

To identify hotspots and forecast crimes:

```python
# Load the cleaned data
data = pd.read_csv("cleaned_crime_data.csv")

# Run DBSCAN clustering
clusters = run_dbscan_clustering(data)

# Forecast crime rates for each hotspot
forecast = run_arima_forecasting(data)
```

---

## **Improvements**

- **Auto ARIMA**: Use `pmdarima` for automatic parameter selection.
- **Short Time Series**: Aggregate by week if daily data is insufficient.
- **Visualization**: Enhance map popups to include plots and summaries.
