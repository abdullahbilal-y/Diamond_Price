# app.py

import streamlit as st
import pandas as pd
import joblib
import os

# Load saved model and label encoders
model = joblib.load("diamond_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")

# App title
st.title("💎 DiamPrice - Diamond Price Estimator")
st.write("Enter diamond features to predict its price.")

# Input form
with st.form("prediction_form"):
    carat = st.number_input("Carat", min_value=0.0, format="%.2f")
    cut = st.selectbox("Cut", label_encoders['cut'].classes_)
    color = st.selectbox("Color", label_encoders['color'].classes_)
    clarity = st.selectbox("Clarity", label_encoders['clarity'].classes_)
    depth = st.number_input("Depth", min_value=0.0, format="%.2f")
    table = st.number_input("Table", min_value=0.0, format="%.2f")
    x = st.number_input("x", min_value=0.0, format="%.2f")
    y = st.number_input("y", min_value=0.0, format="%.2f")
    z = st.number_input("z", min_value=0.0, format="%.2f")
    submit = st.form_submit_button("Predict")

# Handle submission
if submit:
    # Encode categorical features
    cut_encoded = label_encoders['cut'].transform([cut])[0]
    color_encoded = label_encoders['color'].transform([color])[0]
    clarity_encoded = label_encoders['clarity'].transform([clarity])[0]

    input_features = [[carat, cut_encoded, color_encoded, clarity_encoded, depth, table, x, y, z]]
    prediction = model.predict(input_features)[0]
    
    st.success(f"Estimated Diamond Price: ${round(prediction, 2)}")

    # Save to history
    history_data = {
        "Carat": carat, "Cut": cut, "Color": color, "Clarity": clarity,
        "Depth": depth, "Table": table, "x": x, "y": y, "z": z,
        "Predicted Price": round(prediction, 2)
    }
    history_df = pd.DataFrame([history_data])

    if os.path.exists("prediction_history.csv"):
        history_df.to_csv("prediction_history.csv", mode='a', header=False, index=False)
    else:
        history_df.to_csv("prediction_history.csv", index=False)
