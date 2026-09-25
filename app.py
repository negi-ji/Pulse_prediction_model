import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os

# ==============================================================================
# CONFIG & TITLE
# ==============================================================================
st.set_page_config(
    page_title="PulsePredict",
    page_icon="❤️",
    layout="centered"
)

# ==============================================================================
# LOAD TRAINED MODEL
# ==============================================================================
@st.cache_resource
def load_model():
    model_path = "random_forest_pipeline.pkl"
    
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found in the root directory.")
        st.stop()
        
    try:
        # Try loading with joblib first (standard for scikit-learn pipelines)
        model = joblib.load(model_path)
        return model
    except Exception:
        try:
            # Fallback to standard pickle if joblib fails
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            st.error("Failed to load model file. If you used Git LFS, ensure the file isn't stored as an LFS pointer, or re-save the model using joblib.")
            st.exception(e)
            st.stop()

model = load_model()

# ==============================================================================
# HEADER
# ==============================================================================
st.title("❤️ PulsePredict")
st.write("Predict the likelihood of heart disease using clinical indicators.")

st.markdown("---")

# ==============================================================================
# USER INPUT FORM
# ==============================================================================
st.subheader("Patient Clinical Data")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], 
                     format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}[x])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "False" if x == 0 else "True")
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                           format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}[x])

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=230, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2],
                         format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
    ca = st.selectbox("Number of Major Vessels (0-4)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3],
                        format_func=lambda x: {0: "Unknown/Normal", 1: "Fixed Defect", 2: "Normal", 3: "Reversable Defect"}[x])

# Build feature dataframe matching model training schema
input_data = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}])

st.markdown("---")

# ==============================================================================
# PREDICTION LOGIC
# ==============================================================================
if st.button("Predict Pulse/Heart Condition", type="primary", use_container_width=True):
    try:
        prediction = model.predict(input_data)[0]
        
        # If model supports probability outputs
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            confidence = probabilities[prediction] * 100
        else:
            confidence = None

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"🚨 **High Risk Detected**")
            if confidence:
                st.write(f"Model Confidence: **{confidence:.2f}%**")
        else:
            st.success(f"✅ **Low Risk / Normal**")
            if confidence:
                st.write(f"Model Confidence: **{confidence:.2f}%**")
                
    except Exception as e:
        st.error(f"Error during prediction: {e}")
