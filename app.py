
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Irrigation System",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Smart Irrigation System")
st.subheader("Machine Learning Assignment 2")
st.write(
    "This application uses a secondary dataset and a "
    "Random Forest machine learning model to predict pump data "
    "based on soil moisture, temperature, and air humidity."
)

# --------------------------------------------------
# DATASET CONFIGURATION
# --------------------------------------------------

DATA_PATH = (
    Path(__file__).parent /
    "smart_irrigation_secondary_dataset.csv"
)

FEATURES = [
    "Soil Moisture",
    "Temperature",
    "Air Humidity"
]

TARGET = "Pump Data"

# --------------------------------------------------
# DATASET LOADING
# --------------------------------------------------

st.sidebar.header("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload the secondary dataset (CSV)",
    type=["csv"]
)

@st.cache_data
def load_dataset(file_path):
    return pd.read_csv(file_path)

try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Uploaded dataset loaded.")
    elif DATA_PATH.exists():
        df = load_dataset(DATA_PATH)
        st.sidebar.success("Project dataset loaded.")
    else:
        st.error(
            "Dataset not found. Upload your CSV file using "
            "the sidebar or place it in the application folder."
        )
        st.stop()

except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

# --------------------------------------------------
# DATA VALIDATION AND CLEANING
# --------------------------------------------------

required_columns = FEATURES + [TARGET]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "The dataset is missing these required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

df = df[required_columns].copy()

for column in required_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna()

if len(df) < 5:
    st.error(
        "The dataset has too few valid records for model training."
    )
    st.stop()

# Keep the original numeric target labels.
# Do not assume that 0 means OFF or 1 means ON.
if not np.all(df[TARGET] == df[TARGET].astype(int)):
    st.error(
        "Pump Data must contain integer class labels. "
        "Please check the original dataset."
    )
    st.stop()

df[TARGET] = df[TARGET].astype(int)

if df[TARGET].nunique() < 2:
    st.error(
        "The dataset must contain at least two different "
        "Pump Data classes."
    )
    st.stop()

class_counts = df[TARGET].value_counts()

if class_counts.min() < 2:
    st.error(
        "Each Pump Data class needs at least two records "
        "for stratified model evaluation."
    )
    st.stop()

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# --------------------------------------------------
# NAVIGATION TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Dashboard",
    "🔮 Prediction",
    "📊 Dataset",
    "📈 Model Evaluation"
])

# --------------------------------------------------
# TAB 1: DASHBOARD
# --------------------------------------------------

with tab1:

    st.header("System Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Dataset Records",
            len(df)
        )

    with col2:
        st.metric(
            "Model Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Number of Features",
            len(FEATURES)
        )

    st.subheader("Smart Irrigation System Description")

    st.write(
        "The system uses machine learning to study the "
        "relationship between soil moisture, temperature, "
        "air humidity, and pump data. It predicts the "
        "recorded pump class using the Random Forest algorithm."
    )

    st.info(
        "The application displays the original numerical "
        "Pump Data classes (0 and 1). Their ON/OFF meanings "
        "must be verified from the original dataset documentation."
    )

    st.subheader("Dataset Preview")
    st.dataframe(
        df.head(10),
        use_container_width=True
    )

# --------------------------------------------------
# TAB 2: PREDICTION
# --------------------------------------------------

with tab2:

    st.header("Irrigation Pump Prediction")

    st.write(
        "Enter environmental readings to predict the "
        "Pump Data class."
    )

    col1, col2, col3 = st.columns(3)

    input_values = {}

    for column, container in zip(
        FEATURES,
        [col1, col2, col3]
    ):

        with container:

            minimum = float(df[column].min())
            maximum = float(df[column].max())
            average = float(df[column].mean())

            if minimum == maximum:
                input_values[column] = st.number_input(
                    column,
                    value=minimum
                )
            else:
                input_values[column] = st.slider(
                    column,
                    min_value=minimum,
                    max_value=maximum,
                    value=average
                )

    if st.button(
        "Predict Pump Data",
        type="primary"
    ):

        input_df = pd.DataFrame(
            [input_values],
            columns=FEATURES
        )

        prediction = model.predict(input_df)[0]

        probabilities = model.predict_proba(input_df)[0]

        st.subheader("Prediction Result")

        st.success(
            f"Predicted Pump Data class: {prediction}"
        )

        st.write("Prediction probabilities:")

        probability_df = pd.DataFrame({
            "Pump Data Class": model.classes_,
            "Probability (%)": probabilities * 100
        })

        st.dataframe(
            probability_df,
            use_container_width=True
        )

        st.warning(
            "This result represents the numerical class in "
            "the dataset. It should not be interpreted as "
            "Pump ON or Pump OFF until the original label "
            "definitions are verified."
        )

# --------------------------------------------------
# TAB 3: DATASET INFORMATION
# --------------------------------------------------

with tab3:

    st.header("Secondary Dataset Information")

    st.write(
        "This application uses a secondary dataset containing "
        "soil moisture, temperature, air humidity, and pump data."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Number of Records",
            len(df)
        )

    with col2:
        st.metric(
            "Number of Columns",
            len(df.columns)
        )

    st.subheader("Dataset Columns")

    st.write(
        "- Soil Moisture: Measures the moisture level in the soil."
    )

    st.write(
        "- Temperature: Records the surrounding temperature."
    )

    st.write(
        "- Air Humidity: Measures moisture in the air."
    )

    st.write(
        "- Pump Data: Contains the original numerical pump classes."
    )

    st.subheader("Pump Data Class Distribution")

    distribution = (
        df[TARGET]
        .value_counts()
        .sort_index()
        .rename_axis("Pump Data Class")
        .reset_index(name="Number of Records")
    )

    st.dataframe(
        distribution,
        use_container_width=True
    )

    st.bar_chart(
        distribution.set_index("Pump Data Class")
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Dataset Source")

    st.write(
        "Kaur, A., Bhatt, D. P., & Raja, L. (2023). "
        "Soil Moisture, Air Temperature and Humidity, "
        "and Water Motor ON/OFF Monitor Data. Mendeley Data, Version 1."
    )

    st.markdown(
        "[View the original dataset](https://doi.org/10.17632/fpdwmm7nrb.1)"
    )

# --------------------------------------------------
# TAB 4: MODEL EVALUATION
# --------------------------------------------------

with tab4:

    st.header("Random Forest Model Evaluation")

    st.write(
        "The model is evaluated using a test dataset that "
        "was not used during model training."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Training Records",
            len(X_train)
        )

    with col3:
        st.metric(
            "Testing Records",
            len(X_test)
        )

    st.subheader("Confusion Matrix")

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"Actual {label}"
            for label in model.classes_
        ],
        columns=[
            f"Predicted {label}"
            for label in model.classes_
        ]
    )

    st.dataframe(
        matrix_df,
        use_container_width=True
    )

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.subheader("Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    st.dataframe(
        importance_df,
        use_container_width=True
    )

    st.bar_chart(
        importance_df.set_index("Feature")
    )

    st.info(
        "The evaluation results show how the model performed "
        "on this particular test split. They do not guarantee "
        "the same performance in a real irrigation environment."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Machine Learning Assignment 2 | GENML801 | "
    "NDAYIMENYE Samuel (26RP02292) | "
    "HABYARIMANA Stanislas (26RP02060) | "
    "Academic Year 2026/2027"
)
