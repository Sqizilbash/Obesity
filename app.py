# ==============================
# app.py - Obesity Prediction Streamlit App
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# 1️. App title
# -----------------------------
st.title("Obesity Level Prediction App")
st.write("Select input features to predict obesity level using different ML models.")

# -----------------------------
# 2️. Load pickled models
# -----------------------------
model_paths = {
    "Logistic Regression": "model/logistic_model.pkl",
    "Decision Tree": "model/decision_tree_model.pkl",
    "K-Nearest Neighbors": "model/knn_model.pkl",
    "Naive Bayes": "model/naive_bayes_model.pkl",
    "Random Forest": "model/random_forest_model.pkl",
    "XGBoost": "model/xgboost_pipeline.pkl"
}

models = {}
for name, path in model_paths.items():
    models[name] = joblib.load(path)

st.sidebar.header("Choose Model")
selected_model_name = st.sidebar.selectbox("Model", list(models.keys()))
model = models[selected_model_name]

# -----------------------------
# 3️. User Input Form
# -----------------------------
st.sidebar.header("Enter Input Features")

def user_input_features():
    Age = st.sidebar.number_input("Age", 14, 70, 25)
    Height = st.sidebar.number_input("Height (m)", 1.45, 2.0, 1.7, step=0.01)
    Weight = st.sidebar.number_input("Weight (kg)", 30, 200, 70)
    FCVC = st.sidebar.selectbox("Vegetable Consumption (1=Low, 2=Medium, 3=High)", [1,2,3])
    NCP = st.sidebar.selectbox("Number of Main Meals (1-4)", [1,2,3,4])
    CH2O = st.sidebar.slider("Daily Water Consumption (L)", 1.0, 3.0, 2.0, step=0.1)
    FAF = st.sidebar.slider("Physical Activity Frequency", 0, 3, 1)
    TUE = st.sidebar.slider("Time using tech devices", 0, 2, 1)

    Gender = st.sidebar.selectbox("Gender", ["Female","Male"])
    CAEC = st.sidebar.selectbox("Eat between meals?", ["Never","Sometimes","Frequently","Always"])
    CALC = st.sidebar.selectbox("Alcohol Consumption?", ["Never","Sometimes","Frequently","Always"])
    MTRANS = st.sidebar.selectbox("Transportation", ["Automobile","Bike","Motorbike","Public_Transportation","Walking"])

    FAVC = st.sidebar.selectbox("Eat High Calorie Food?", [0,1])
    SMOKE = st.sidebar.selectbox("Smoke?", [0,1])
    SCC = st.sidebar.selectbox("Monitor Calories?", [0,1])
    family_history_with_overweight = st.sidebar.selectbox("Family history of overweight?", [0,1])

    data = {
        "Age": Age,
        "Height": Height,
        "Weight": Weight,
        "FCVC": FCVC,
        "NCP": NCP,
        "CH2O": CH2O,
        "FAF": FAF,
        "TUE": TUE,
        "Gender": Gender,
        "CAEC": CAEC,
        "CALC": CALC,
        "MTRANS": MTRANS,
        "FAVC": FAVC,
        "SMOKE": SMOKE,
        "SCC": SCC,
        "family_history_with_overweight": family_history_with_overweight
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# -----------------------------
# 4.Make Prediction
# -----------------------------
if st.button("Predict Obesity Level"):

    # Predict
    prediction = model.predict(input_df)
    
    # Some pipelines output arrays (like XGBoost pipeline)
    if isinstance(prediction, np.ndarray):
        prediction = prediction[0]

    st.subheader("Predicted Obesity Level:")
    st.success(prediction)

    # Optional: probability if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)
        st.subheader("Prediction Probabilities:")
        proba_df = pd.DataFrame(proba, columns=model.classes_)
        st.dataframe(proba_df)