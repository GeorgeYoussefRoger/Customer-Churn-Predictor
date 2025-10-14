import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("models/final_model.keras", compile=False)
scaler = joblib.load("models/scaler.pkl")

st.title("📊 Customer Churn Predictor")

# Input fields
st.write("Enter customer information below.")

gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
online_security = st.selectbox("Online Security", ["No", "Yes"])
online_backup = st.selectbox("Online Backup", ["No", "Yes"])
device_protection = st.selectbox("Device Protection", ["No", "Yes"])
tech_support = st.selectbox("Tech Support", ["No", "Yes"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=2500.0)

internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method",
                              ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])

# Convert inputs to DataFrame
input_data = pd.DataFrame({
    'gender': [1 if gender == 'Female' else 0],
    'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
    'Partner': [1 if partner == 'Yes' else 0],
    'Dependents': [1 if dependents == 'Yes' else 0],
    'tenure': [tenure],
    'PhoneService': [1 if phone_service == 'Yes' else 0],
    'MultipleLines': [1 if multiple_lines == 'Yes' else 0],
    'OnlineSecurity': [1 if online_security == 'Yes' else 0],
    'OnlineBackup': [1 if online_backup == 'Yes' else 0],
    'DeviceProtection': [1 if device_protection == 'Yes' else 0],
    'TechSupport': [1 if tech_support == 'Yes' else 0],
    'StreamingTV': [1 if streaming_tv == 'Yes' else 0],
    'StreamingMovies': [1 if streaming_movies == 'Yes' else 0],
    'PaperlessBilling': [1 if paperless_billing == 'Yes' else 0],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'InternetService_0': [1 if internet_service == 'No' else 0],
    'InternetService_DSL': [1 if internet_service == 'DSL' else 0],
    'InternetService_Fiber optic': [1 if internet_service == 'Fiber optic' else 0],
    'Contract_Month-to-month': [1 if contract == 'Month-to-month' else 0],
    'Contract_One year': [1 if contract == 'One year' else 0],
    'Contract_Two year': [1 if contract == 'Two year' else 0],
    'PaymentMethod_Bank transfer (automatic)': [1 if payment_method == 'Bank transfer (automatic)' else 0],
    'PaymentMethod_Credit card (automatic)': [1 if payment_method == 'Credit card (automatic)' else 0],
    'PaymentMethod_Electronic check': [1 if payment_method == 'Electronic check' else 0],
    'PaymentMethod_Mailed check': [1 if payment_method == 'Mailed check' else 0],
})

# Scale numeric features
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
input_data[numeric_features] = scaler.transform(input_data[numeric_features])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0][0] > 0.5:
        st.error("⚠️ The customer is likely to churn")
    else:
        st.success("✅ The customer is not likely to churn")
