
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(
    page_title="Smart Irrigation System",
    page_icon="🌱",
    layout="centered"
)

# Application title
st.title("🌱 Smart Irrigation System")
st.write(
    "A machine-learning system that predicts whether "
    "irrigation may be needed using environmental measurements."
)

# Train the Random Forest model
@st.cache_resource
def train_model():
    rng = np.random.default_rng(42)
    n = 3000

    # Generate simulated training data
    moisture = rng.uniform(5, 90, n)
    temperature = rng.uniform(10, 45, n)
    humidity = rng.uniform(15, 100, n)

    # Simulated irrigation decision rule
    moisture_limit = (
        35
        + np.maximum(temperature - 25, 0) * 0.5
        - np.maximum(humidity - 70, 0) * 0.12
    )

    labels = (moisture < moisture_limit).astype(int)

    # Add small variation to training labels
    noise = rng.random(n) < 0.04
    labels[noise] = 1 - labels[noise]

    X = pd.DataFrame({
        "Soil Moisture": moisture,
        "Temperature": temperature,
        "Air Humidity": humidity
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    test_accuracy = accuracy_score(
        y_test,
        model.predict(X_test)
    )

    return model, test_accuracy


model, test_accuracy = train_model()

# Input measurements
st.subheader("Enter Environmental Measurements")

soil_moisture = st.slider(
    "Soil Moisture (%)",
    min_value=0,
    max_value=100,
    value=15
)

temperature = st.slider(
    "Temperature (°C)",
    min_value=0,
    max_value=50,
    value=32
)

air_humidity = st.slider(
    "Air Humidity (%)",
    min_value=0,
    max_value=100,
    value=40
)

# Make a prediction
if st.button("Predict Irrigation", type="primary"):
    input_data = pd.DataFrame([{
        "Soil Moisture": soil_moisture,
        "Temperature": temperature,
        "Air Humidity": air_humidity
    }])

    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]
    confidence = probabilities[prediction] * 100

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("💧 IRRIGATE")
        st.write(
            "The model predicts that irrigation may be needed."
        )
    else:
        st.success("🌱 DO NOT IRRIGATE")
        st.write(
            "The model predicts that irrigation may not be needed."
        )

    st.metric(
        "Model Confidence",
        f"{confidence:.1f}%"
    )

    st.caption(
        "This prototype uses simulated training data. "
        "Model confidence does not guarantee that a real crop "
        "needs or does not need water. Validate with real farm "
        "measurements before relying on these predictions."
    )

# Display model information
st.divider()
st.subheader("Model Information")

st.write("Algorithm: Random Forest Classification")
st.write("Training examples: 3,000 simulated measurements")
st.write(
    f"Test accuracy on simulated data: {test_accuracy:.1%}"
)

# Display current measurements
st.subheader("Current Measurements")

measurements = pd.DataFrame({
    "Measurement": [
        "Soil Moisture",
        "Temperature",
        "Air Humidity"
    ],
    "Value": [
        f"{soil_moisture}%",
        f"{temperature} °C",
        f"{air_humidity}%"
    ]
})

st.dataframe(
    measurements,
    hide_index=True,
    use_container_width=True
)
