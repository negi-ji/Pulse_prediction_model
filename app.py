import streamlit as st
import pandas as pd
import pickle
import streamlit as st


@st.cache_resource
def load_model():
    with open("random_forest_pipeline.pkl", "rb") as f:
        model = pickle.load(f)
    return model


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PulsePredict",
    page_icon="❤️",
    layout="wide"
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    with open("random_forest_pipeline.pkl", "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()


# =========================================================
# HEADER
# =========================================================

st.title("❤️ PulsePredict")

st.subheader("Clinical Risk Analysis Prototype")

st.write(
    """
    Enter the patient information below to generate
    a machine-learning risk prediction.
    """
)

st.warning(
    """
    ⚠️ Educational prototype only.
    This application is not a medical diagnostic tool
    and should not be used for clinical decision-making.
    """
)


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.header("Patient Information")


col1, col2, col3 = st.columns(3)


# =========================================================
# COLUMN 1
# =========================================================

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50,
        step=1
    )

    sex_label = st.selectbox(
        "Sex",
        [
            "Female",
            "Male"
        ]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120,
        step=1
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=600,
        value=200,
        step=1
    )


# =========================================================
# COLUMN 2
# =========================================================

with col2:

    cp_label = st.selectbox(
        "Chest Pain Type",
        [
            "Type 1",
            "Type 2",
            "Type 3",
            "Type 4"
        ]
    )

    fbs_label = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [
            "No",
            "Yes"
        ]
    )

    restecg_label = st.selectbox(
        "Resting ECG",
        [
            "Type 0",
            "Type 1",
            "Type 2"
        ]
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150,
        step=1
    )


# =========================================================
# COLUMN 3
# =========================================================

with col3:

    exang_label = st.selectbox(
        "Exercise-Induced Angina",
        [
            "No",
            "Yes"
        ]
    )

    oldpeak = st.number_input(
        "ST Depression",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope_label = st.selectbox(
        "ST Segment Slope",
        [
            "Type 1",
            "Type 2",
            "Type 3"
        ]
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        [
            0,
            1,
            2,
            3
        ]
    )

    thal = st.selectbox(
        "Thalassemia Category",
        [
            3,
            6,
            7
        ]
    )


# =========================================================
# CONVERT USER-FRIENDLY VALUES TO DATASET VALUES
# =========================================================

sex = {
    "Female": 0,
    "Male": 1
}[sex_label]


cp = {
    "Type 1": 1,
    "Type 2": 2,
    "Type 3": 3,
    "Type 4": 4
}[cp_label]


fbs = {
    "No": 0,
    "Yes": 1
}[fbs_label]


restecg = {
    "Type 0": 0,
    "Type 1": 1,
    "Type 2": 2
}[restecg_label]


exang = {
    "No": 0,
    "Yes": 1
}[exang_label]


slope = {
    "Type 1": 1,
    "Type 2": 2,
    "Type 3": 3
}[slope_label]


# =========================================================
# CREATE PATIENT DATAFRAME
# =========================================================

patient = pd.DataFrame({

    "age": [age],

    "sex": [sex],

    "cp": [cp],

    "trestbps": [trestbps],

    "chol": [chol],

    "fbs": [fbs],

    "restecg": [restecg],

    "thalach": [thalach],

    "exang": [exang],

    "oldpeak": [oldpeak],

    "slope": [slope],

    "ca": [ca],

    "thal": [thal]

})


# =========================================================
# OPTIONAL: VIEW INTERNAL MODEL INPUT
# =========================================================

with st.expander("View model input"):

    st.dataframe(
        patient,
        use_container_width=True
    )


# =========================================================
# ANALYZE BUTTON
# =========================================================

st.divider()

if st.button(
    "🔍 Analyze Patient",
    use_container_width=True
):

    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    prediction = model.predict(patient)[0]

    probability = model.predict_proba(
        patient
    )[0][1]


    # =====================================================
    # RESULT SECTION
    # =====================================================

    st.header("Prediction Result")

    result_col1, result_col2 = st.columns(2)


    # -----------------------------------------------------
    # PROBABILITY
    # -----------------------------------------------------

    with result_col1:

        st.metric(
            "Predicted Risk Probability",
            f"{probability * 100:.2f}%"
        )


    # -----------------------------------------------------
    # YES / NO RESULT
    # -----------------------------------------------------

    with result_col2:

        if prediction == 1:

            st.error(
                "Prediction: YES"
            )

        else:

            st.success(
                "Prediction: NO"
            )


    # =====================================================
    # PROGRESS BAR
    # =====================================================

    st.subheader("Risk Probability")

    st.progress(
        float(probability)
    )


    # =====================================================
    # RESULT EXPLANATION
    # =====================================================

    if prediction == 1:

        st.write(
            """
            The trained model predicts the positive class
            for this patient input.
            """
        )

    else:

        st.write(
            """
            The trained model predicts the negative class
            for this patient input.
            """
        )


    # =====================================================
    # IMPORTANT DISCLAIMER
    # =====================================================

    st.info(
        f"""
        The model produced a predicted probability of
        **{probability * 100:.2f}% for the positive class**.

        This value comes from the trained machine-learning
        model and has not been clinically validated for
        individual patient diagnosis.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "PulsePredict V1 | Educational Machine Learning Prototype"
)
