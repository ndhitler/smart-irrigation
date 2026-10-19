
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Machine Learning Assignment 2",
    page_icon="🌱",
    layout="centered"
)

# ---------------- STUDENT INFORMATION ----------------

if "show_prediction_app" not in st.session_state:
    st.session_state.show_prediction_app = False

if not st.session_state.show_prediction_app:
    st.title("RP Kigali College")
    st.subheader("Student Information")
    st.markdown("---")
    
    st.subheader("**Group Members:**")
    st.markdown("**Name:** NDAYIMENYE Samuel - 26RP02292")
    st.markdown("**Name:** HABYARIMANA Stanislas - 26RP02060")

    st.markdown("---")

    st.markdown(
        "**Program:** BTech in Electronics and Telecommunication Technology"
    )
    st.markdown("**Institution:** RP Kigali College")
    st.markdown("**Course/Module:** Machine Learning")
    st.markdown("**Module Code:** GENML801")
    st.markdown("**Assignment:** Machine Learning Assignment 2")

    st.markdown("---")

    st.info(
        "Project Title: Smart Irrigation System "
        "Using Random Forest Classification"
    )

    if st.button("Continue to Smart Irrigation System", type="primary"):
        st.session_state.show_prediction_app = True
        st.rerun()

    st.stop()

# ---------------- SMART IRRIGATION SYSTEM ----------------

st.title("🌱 Smart Irrigation System")
st.write(
    "This application uses a Random Forest machine learning model "
    "to predict whether irrigation is needed."
)

if st.button("← Back to Student Information"):
    st.session_state.show_prediction_app = False
    st.rerun()

st.markdown("---")

# Generate simulated training data
@st.cache_resource
def train_model():
    np.random.seed(42)

    n_samples = 3000

    soil_moisture = np.random.uniform(5, 90, n_samples)
    temperature = np.random.uniform(15, 45, n_samples)
    air_humidity = np.random.uniform(20, 95, n_samples)

    # Simulated rule for creating training labels
    irrigation_needed = (
        (soil_moisture < 35)
        & ((temperature > 25) | (air_humidity < 50))
    )

    data = pd.DataFrame({
        "Soil Moisture": soil_moisture,
        "Temperature": temperature,
        "Air Humidity": air_humidity,
        "Irrigation Needed": irrigation_needed.astype(int)
    })

    X = data[
        ["Soil Moisture", "Temperature", "Air Humidity"]
    ]
    y = data["Irrigation Needed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, accuracy

model, accuracy = train_model()

st.success("Machine learning model is ready.")

st.subheader("Enter Environmental Conditions")

soil = st.slider(
    "Soil Moisture (%)", min_value=5, max_value=90, value=30
)

temperature = st.slider(
    "Temperature (°C)", min_value=15, max_value=42, value=30
)

humidity = st.slider(
    "Air Humidity (%)", min_value=20, max_value=95, value=45
)

if st.button("Predict Irrigation", type="primary"):
    input_data = pd.DataFrame({
        "Soil Moisture": [soil],
        "Temperature": [temperature],
        "Air Humidity": [humidity]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("💧 IRRIGATE — The model predicts that irrigation is needed.")
    else:
        st.success("🌿 DO NOT IRRIGATE — The model predicts that irrigation is not needed.")

    st.caption(
        "This prediction is based on simulated training data, "
        "not live soil sensors."
    )

st.markdown("---")
st.subheader("Model Information")
st.write("**Algorithm:** Random Forest Classification")
st.write("**Training data:** 3,000 simulated records")
st.write(f"**Test accuracy on simulated data:** {accuracy:.2%}")

)

st.markdown("---")
st.caption(
    "NDAYIMENYE Samuel - 26RP02292 | HABYARIMANA Stanislas - 26RP02060 | GENML801 | "
    "Machine Learning Assignment 2"
)
