import streamlit as st
import pandas as pd
import joblib

# Load pipeline
pipeline = joblib.load("models/final_pipeline.pkl")

st.title("📉 Customer Churn Predictor")
st.write("Enter customer information.")

col1, col2, col3 = st.columns(3)

# Customer Info
with col1:
    st.subheader("Customer Info")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)

# Services
with col2:
    st.subheader("Services")
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = "No"
    if phone_service == "Yes":
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
    online_security = st.selectbox("Online Security", ["No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes"])
    streaming_options = st.multiselect("Streaming Services", ["Streaming TV", "Streaming Movies"])
    
# Billing & Plan
with col3:
    st.subheader("Billing & Plan")
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=2500.0)
    internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", 
                                                     "Mailed check"])

# Convert inputs to DataFrame
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'Dependents': [dependents],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': ["Yes" if "Streaming TV" in streaming_options else "No"],
    'StreamingMovies': ["Yes" if "Streaming Movies" in streaming_options else "No"],
    'PaperlessBilling': [paperless_billing],
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'InternetService': [internet_service],
    'Contract': [contract],
    'PaymentMethod': [payment_method]
})

# Predict
if st.button("Predict"):
    if total_charges < monthly_charges * tenure:
        st.warning("Total charges seem inconsistent with tenure.")
        st.stop()

    proba = pipeline.predict_proba(input_data)[0][1]

    if proba >= 0.6:
        st.error(f"High Churn Risk ({proba:.2%})")
    elif proba >= 0.3:
        st.warning(f"Moderate Churn Risk ({proba:.2%})")
    else:
        st.success(f"Low Churn Risk ({proba:.2%})")