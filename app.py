
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Disease Research Demo",
    page_icon="❤️",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load(
        "heart_disease_catboost_pipeline.joblib"
    )

model = load_model()

st.title("Heart Disease Detection")
st.write(
    "Research demonstration using a trained "
    "CatBoost machine-learning pipeline."
)

st.warning(
    "Research use only. This tool is not a "
    "medical diagnostic device."
)

with st.form("prediction_form"):
    age = st.number_input(
        "Age", 1, 120, 50
    )

    sex = st.selectbox(
        "Sex", ["Male", "Female"]
    )

    cp = st.selectbox(
        "Chest pain type",
        ["typical angina", "atypical angina",
         "non-anginal", "asymptomatic"]
    )

    trestbps = st.number_input(
        "Resting blood pressure (mm Hg)",
        50, 250, 120
    )

    chol = st.number_input(
        "Cholesterol (mg/dl)",
        0, 700, 200
    )

    fbs = st.selectbox(
        "Fasting blood sugar > 120 mg/dl",
        ["False", "True"]
    )

    restecg = st.selectbox(
        "Resting ECG",
        ["normal", "stt abnormality",
         "lv hypertrophy"]
    )

    thalch = st.number_input(
        "Maximum heart rate achieved",
        40, 250, 150
    )

    exang = st.selectbox(
        "Exercise-induced angina",
        ["False", "True"]
    )

    oldpeak = st.number_input(
        "ST depression",
        0.0, 10.0, 1.0, step=0.1
    )

    slope = st.selectbox(
        "ST segment slope",
        ["upsloping", "flat", "downsloping"]
    )

    ca = st.selectbox(
        "Number of major vessels",
        [0, 1, 2, 3, 4]
    )

    thal = st.selectbox(
        "Thalassemia",
        ["normal", "fixed defect",
         "reversible defect"]
    )

    submitted = st.form_submit_button(
        "Run model prediction"
    )

if submitted:
    row = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    try:
        prediction = int(
            model.predict(row).ravel()[0]
        )
        score = model.predict_proba(row)[0, 1]

        st.subheader("Model output")
        st.write("Predicted class:", prediction)
        st.write(
            f"Class 1 score: {score:.4f}"
        )

        st.info(
            "This result is a model output, not a "
            "diagnosis. Consult a qualified clinician "
            "for medical concerns."
        )

    except Exception as error:
        st.error(
            "Prediction failed. Check the input "
            "categories against the training dataset."
        )
        st.code(str(error))
