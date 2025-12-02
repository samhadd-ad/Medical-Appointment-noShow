"""
Streamlit UI application for the Medical Appointment No-Show prediction.

This app:
- Loads the trained scaler, best model, and feature column names
- Lets a user input patient & appointment information
- Outputs the predicted probability of a No-show
- Shows a sample of the dataset

To run:
    streamlit run app.py
"""

import pickle

import pandas as pd
import streamlit as st


DATA_FILE = "KaggleV2-May-2016.csv"
SCALER_FILE = "scaler.pkl"
MODEL_FILES = {
    "Logistic Regression": "models/best_model.pkl",
    "Decision Tree": "models/model_dt.pkl",
    "Random Forest": "models/model_rf.pkl",
}
FEATURE_COLS_FILE = "feature_columns.pkl"


@st.cache_data
def load_raw_data(path: str = DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Basic cleaning consistent with analysis.py
    df = df[df["Age"] >= 0]
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
    df["WaitingDays"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days
    df = df[df["WaitingDays"] >= 0]

    return df


@st.cache_resource
def load_artifacts():
    # Load scaler
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    # Load feature columns
    with open(FEATURE_COLS_FILE, "rb") as f:
        feature_cols = pickle.load(f)
    # Load all models
    models_dict = {}
    for name, path in MODEL_FILES.items():
        with open(path, "rb") as f:
            models_dict[name] = pickle.load(f)
    return scaler, feature_cols, models_dict

raw_df = load_raw_data()
scaler, feature_columns, models_dict = load_artifacts()

# Prepare lists for dropdowns
neighbourhoods = sorted(raw_df["Neighbourhood"].unique().tolist())
genders = ["F", "M"]


def build_feature_row(
    age: int,
    gender: str,
    neighbourhood: str,
    scholarship: int,
    hypertension: int,
    diabetes: int,
    alcoholism: int,
    handcap: int,
    sms_received: int,
    waiting_days: int,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame with the same feature structure
    as used during training (including one-hot encoding).
    """
    base_dict = {
        "Age": age,
        "Scholarship": scholarship,
        "Hipertension": hypertension,  # note the original spelling
        "Diabetes": diabetes,
        "Alcoholism": alcoholism,
        "Handcap": handcap,
        "SMS_received": sms_received,
        "WaitingDays": waiting_days,
        "Gender": gender,
        "Neighbourhood": neighbourhood,
    }

    row = pd.DataFrame([base_dict])

    # One-hot encode categorical features as in analysis.py
    row_encoded = pd.get_dummies(
        row,
        columns=["Gender", "Neighbourhood"],
        drop_first=True,
    )

    # Align columns with training feature columns
    row_encoded = row_encoded.reindex(columns=feature_columns, fill_value=0)

    return row_encoded


# ================== Streamlit UI Layout ==================

st.title("Medical Appointment No-Show Predictor")
st.write(
    """
This application is part of the ITEC 3040 Final Project (Group G).

Use the controls in the sidebar to specify patient and appointment information.
The model will estimate the probability that the patient will **miss** the appointment
(No-show = Yes).
"""
)

st.sidebar.header("Input Features")

# Sidebar inputs

selected_model_name = st.sidebar.selectbox(
    "Select model to use for prediction",
    list(MODEL_FILES.keys())
)
model = models_dict[selected_model_name]
age = st.sidebar.slider(
    "Age",
    int(raw_df["Age"].min()),
    int(raw_df["Age"].max()),
    30,
)

gender = st.sidebar.selectbox("Gender", genders)

neighbourhood = st.sidebar.selectbox("Neighbourhood", neighbourhoods)

scholarship = st.sidebar.selectbox("Scholarship (Bolsa Família)", [0, 1])

hypertension = st.sidebar.selectbox("Hypertension", [0, 1])

diabetes = st.sidebar.selectbox("Diabetes", [0, 1])

alcoholism = st.sidebar.selectbox("Alcoholism", [0, 1])

handcap = st.sidebar.selectbox("Handcap (0–4)", [0, 1, 2, 3, 4])

sms_received = st.sidebar.selectbox("SMS received", [0, 1])

waiting_days = st.sidebar.slider(
    "Waiting days between scheduling and appointment",
    int(raw_df["WaitingDays"].min()),
    int(min(60, raw_df["WaitingDays"].max())),
    5,
)

st.write("### Current Input Summary")
st.write(f"- **Model**: {selected_model_name}")
st.write(f"- **Age**: {age}")
st.write(f"- **Gender**: {gender}")
st.write(f"- **Neighbourhood**: {neighbourhood}")
st.write(f"- **WaitingDays**: {waiting_days}")
st.write(
    f"- **Scholarship**: {scholarship}, "
    f"Hypertension: {hypertension}, Diabetes: {diabetes}, Alcoholism: {alcoholism}"
)
st.write(f"- **Handcap**: {handcap}, SMS_received: {sms_received}")

if st.button("Predict No-show Probability"):
    X_new = build_feature_row(
        age=age,
        gender=gender,
        neighbourhood=neighbourhood,
        scholarship=scholarship,
        hypertension=hypertension,
        diabetes=diabetes,
        alcoholism=alcoholism,
        handcap=handcap,
        sms_received=sms_received,
        waiting_days=waiting_days,
    )

    # Scale only for Logistic Regression
    if selected_model_name == "Logistic Regression":
        X_input = scaler.transform(X_new)
    else:
        X_input = X_new

    # Predict probability
    if hasattr(model, "predict_proba"):
        proba_no_show = model.predict_proba(X_input)[0, 1]
    else:
        pred_label = model.predict(X_input)[0]
        proba_no_show = float(pred_label)

    st.subheader(f"Prediction Result ({selected_model_name})")
    st.write(
        f"Estimated probability of **No-show** (miss appointment): "
        f"**{proba_no_show * 100:.1f}%**"
    )
    st.caption(f"(Raw model output: {proba_no_show:.4f})")

    if proba_no_show >= 0.5:
        st.write(
            "The model suggests this patient is **more likely to miss** the appointment."
        )
    else:
        st.write(
            "The model suggests this patient is **more likely to attend** the appointment."
        )

st.write("### Sample of Cleaned Dataset (first 10 rows)")
st.dataframe(raw_df.head(10))