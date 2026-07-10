import streamlit as st
import numpy as np
import joblib
model = joblib.load(r"models\random_forest_model.pkl")
scaler = joblib.load(r"models\scaler.pkl")
st.title("Customer Purchase Prediction")
age = st.number_input("Age", min_value=18, max_value=100)
salary = st.number_input(
    "Estimated Salary",
    min_value=10000,
    max_value=200000
)
if st.button("Predict"):
    data = np.array([[age, salary]])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    if prediction[0] == 1:
        st.success("Customer likely to purchase")
    else:
        st.error("Customer unlikely to purchase")
st.write("Model being used is Random Forest")
st.write("Accuracy : 92.5%")