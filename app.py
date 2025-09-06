import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("bike_model.pkl")

st.set_page_config(page_title="Bike Rental Prediction", layout="centered")

# Title
st.title("Bike Rental Prediction App")
st.write(
    "This application uses a trained Machine Learning model to predict the number "
    "of bike rentals based on weather conditions, season, and calendar features."
)

# Sidebar inputs
st.sidebar.header("Input Features")

season = st.sidebar.selectbox("Season (1: Spring, 2: Summer, 3: Fall, 4: Winter)", [1, 2, 3, 4])
yr = st.sidebar.selectbox("Year (0: 2011, 1: 2012)", [0, 1])
mnth = st.sidebar.slider("Month (1-12)", 1, 12, 6)
holiday = st.sidebar.selectbox("Holiday (0: No, 1: Yes)", [0, 1])
weekday = st.sidebar.slider("Weekday (0 = Sunday ... 6 = Saturday)", 0, 6, 2)
workingday = st.sidebar.selectbox("Working Day (0: No, 1: Yes)", [0, 1])
weathersit = st.sidebar.selectbox(
    "Weather Situation (1: Clear, 2: Mist/Cloudy, 3: Light Snow/Rain, 4: Heavy Rain)",
    [1, 2, 3, 4]
)

# --- Weather Scenario Shortcut ---
st.sidebar.subheader("Quick Weather Scenario")
scenario = st.sidebar.selectbox(
    "Select a scenario (optional)",
    ["Custom", "Sunny & Warm", "Cloudy & Mild", "Rainy Day", "Cold Winter Day"]
)

if scenario == "Sunny & Warm":
    temp_c, atemp_c, hum_percent, windspeed_kmh = 28, 30, 40, 10
elif scenario == "Cloudy & Mild":
    temp_c, atemp_c, hum_percent, windspeed_kmh = 20, 21, 60, 15
elif scenario == "Rainy Day":
    temp_c, atemp_c, hum_percent, windspeed_kmh = 18, 18, 80, 20
elif scenario == "Cold Winter Day":
    temp_c, atemp_c, hum_percent, windspeed_kmh = 5, 4, 55, 12
else:
    # Manual input if "Custom" is chosen
    temp_c = st.sidebar.slider("Temperature (°C)", -5, 40, 20)
    atemp_c = st.sidebar.slider("Feels-like Temperature (°C)", -5, 50, 22)
    hum_percent = st.sidebar.slider("Humidity (%)", 0, 100, 60)
    windspeed_kmh = st.sidebar.slider("Windspeed (km/h)", 0, 67, 15)

# Convert to normalized values (as in dataset)
temp = temp_c / 41
atemp = atemp_c / 50
hum = hum_percent / 100
windspeed = windspeed_kmh / 67

# Input data for model
input_data = pd.DataFrame({
    "season": [season],
    "yr": [yr],
    "mnth": [mnth],
    "holiday": [holiday],
    "weekday": [weekday],
    "workingday": [workingday],
    "weathersit": [weathersit],
    "temp": [temp],
    "atemp": [atemp],
    "hum": [hum],
    "windspeed": [windspeed]
})

# Prediction
prediction = model.predict(input_data)[0]

# Display prediction
st.subheader("Predicted Bike Rentals")
st.metric(label="Expected Number of Rentals", value=int(prediction))

# Conditional messages
if prediction > 6000:
    st.success("High demand expected. Prepare for a very busy day.")
elif prediction > 3000:
    st.info("Moderate demand expected. A regular day ahead.")
else:
    st.warning("Low demand expected. Likely a quiet day.")

# Show input features
st.write("### Input Features Used (Normalized for Model)")
st.write(input_data)

# Visualization
st.write("### Visualization of Prediction")
fig, ax = plt.subplots()
ax.bar(["Predicted Rentals"], [prediction], color="skyblue")
ax.set_ylabel("Number of Rentals")
st.pyplot(fig)

# About section
st.write("---")
st.write(
    "**About this Project:** This application is built on the Bike Rental dataset "
    "using a Linear Regression model. It predicts bike rental demand based on "
    "features such as season, weather conditions, temperature, and working days."
)
