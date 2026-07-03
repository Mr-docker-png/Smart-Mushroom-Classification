import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🍄 Smart Mushroom Classification",
    page_icon="🍄",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("models/mushroom_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
model_columns = joblib.load("models/model_columns.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("🍄 Smart Mushroom Classification")

st.markdown(
    """
Predict whether a mushroom is **Edible** or **Poisonous**
using a Machine Learning model built with **Support Vector Machine (SVM)**.
"""
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📊 Model Information")

st.sidebar.success("Algorithm : Support Vector Machine")

st.sidebar.write("Accuracy : **100%**")

st.sidebar.write("Dataset Size : **8124 Mushrooms**")

st.sidebar.write("Features : **22**")

st.sidebar.write("Encoding : **One-Hot Encoding**")

st.sidebar.write("Scaling : **StandardScaler**")

st.sidebar.write("---")

st.sidebar.info(
"""
This application predicts whether a mushroom is edible or poisonous using a trained Support Vector Machine model.
"""
)

# -----------------------------
# Input Section
# -----------------------------
st.header("🍄 Mushroom Characteristics")

col1, col2 = st.columns(2)

with col1:

    cap_shape = st.selectbox(
        "Cap Shape",
        [
            "b","c","x","f","k","s"
        ]
    )

    cap_surface = st.selectbox(
        "Cap Surface",
        [
            "f","g","y","s"
        ]
    )

    cap_color = st.selectbox(
        "Cap Color",
        [
            "n","b","c","g","r","p","u","e","w","y"
        ]
    )

    bruises = st.selectbox(
        "Bruises",
        [
            "t","f"
        ]
    )

    odor = st.selectbox(
        "Odor",
        [
            "a","l","c","y","f","m","n","p","s"
        ]
    )

    gill_attachment = st.selectbox(
        "Gill Attachment",
        [
            "a","f"
        ]
    )

    gill_spacing = st.selectbox(
        "Gill Spacing",
        [
            "c","w"
        ]
    )

    gill_size = st.selectbox(
        "Gill Size",
        [
            "b","n"
        ]
    )

    gill_color = st.selectbox(
        "Gill Color",
        [
            "k","n","b","h","g","r","o","p","u","e","w","y"
        ]
    )

    stalk_shape = st.selectbox(
        "Stalk Shape",
        [
            "e","t"
        ]
    )

with col2:

    stalk_root = st.selectbox(
        "Stalk Root",
        [
            "b","c","u","e","z"
        ]
    )

    stalk_surface_above_ring = st.selectbox(
        "Stalk Surface Above Ring",
        [
            "f","y","k","s"
        ]
    )

    stalk_surface_below_ring = st.selectbox(
        "Stalk Surface Below Ring",
        [
            "f","y","k","s"
        ]
    )

    stalk_color_above_ring = st.selectbox(
        "Stalk Color Above Ring",
        [
            "n","b","c","g","o","p","e","w","y"
        ]
    )

    stalk_color_below_ring = st.selectbox(
        "Stalk Color Below Ring",
        [
            "n","b","c","g","o","p","e","w","y"
        ]
    )

    veil_color = st.selectbox(
        "Veil Color",
        [
            "n","o","w","y"
        ]
    )

    ring_number = st.selectbox(
        "Ring Number",
        [
            "n","o","t"
        ]
    )

    ring_type = st.selectbox(
        "Ring Type",
        [
            "c","e","f","l","n","p","s","z"
        ]
    )

    spore_print_color = st.selectbox(
        "Spore Print Color",
        [
            "k","n","b","h","r","o","u","w","y"
        ]
    )

    population = st.selectbox(
        "Population",
        [
            "a","c","n","s","v","y"
        ]
    )

    habitat = st.selectbox(
        "Habitat",
        [
            "g","l","m","p","u","w","d"
        ]
    )

# -----------------------------
# Predict Button
# -----------------------------
predict_button = st.button(
    "🔍 Predict Mushroom"
)

# -----------------------------
# Prediction
# -----------------------------
if predict_button:

    input_data = pd.DataFrame({
        "cap-shape":[cap_shape],
        "cap-surface":[cap_surface],
        "cap-color":[cap_color],
        "bruises":[bruises],
        "odor":[odor],
        "gill-attachment":[gill_attachment],
        "gill-spacing":[gill_spacing],
        "gill-size":[gill_size],
        "gill-color":[gill_color],
        "stalk-shape":[stalk_shape],
        "stalk-root":[stalk_root],
        "stalk-surface-above-ring":[stalk_surface_above_ring],
        "stalk-surface-below-ring":[stalk_surface_below_ring],
        "stalk-color-above-ring":[stalk_color_above_ring],
        "stalk-color-below-ring":[stalk_color_below_ring],
        "veil-color":[veil_color],
        "ring-number":[ring_number],
        "ring-type":[ring_type],
        "spore-print-color":[spore_print_color],
        "population":[population],
        "habitat":[habitat]
    })

    # -----------------------------
    # One-Hot Encoding
    # -----------------------------
    input_data = pd.get_dummies(input_data)

    # -----------------------------
    # Match Training Columns
    # -----------------------------
    input_data = input_data.reindex(
        columns=model_columns,
        fill_value=0
    )

    # -----------------------------
    # Scaling
    # -----------------------------
    input_scaled = scaler.transform(input_data)

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict(input_scaled)
    
    decision_score = model.decision_function(input_scaled)

    confidence = abs(decision_score[0])

    confidence = min(confidence*20,100)

    prediction = label_encoder.inverse_transform(prediction)

    st.write("---")

    st.subheader("Prediction Result")

    if prediction[0] == "e":

        st.success("🟢 Edible Mushroom")

        st.info(
            """
            This mushroom is predicted to be **Edible**.

            The prediction is based on the characteristics you selected.
            """
        )

    else:

        st.error("🔴 Poisonous Mushroom")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


        st.warning(
            """
            This mushroom is predicted to be **Poisonous**.

            **Do not consume this mushroom.**

            Many poisonous mushrooms looks similar to edible spicies.
            """
        )
st.markdown("""
<style>

.main{
    padding-top:20px;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    font-size:20px;
    font-weight:bold;
}

.result-box{
    padding:25px;
    border-radius:15px;
    margin-top:20px;
    text-align:center;
}

.footer{
    text-align:center;
    margin-top:40px;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

st.write("---")

st.header("About This Project")

st.write("""
This project was developed using **Support Vector Machine (SVM)**.

### Algorithms Compared

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine

### Final Model

Support Vector Machine

### Dataset

8124 Mushroom Samples

### Features

22 Original Features

116 Features After One-Hot Encoding

### Accuracy

100%
""")

st.write("---")

st.markdown(
"""
<div class='footer'>

Developed with ❤️ using Python, Scikit-learn & Streamlit

</div>
""",
unsafe_allow_html=True
)