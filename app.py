import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('credit_risk.pkl')

# Streamlit app title
st.title("Credit Risk Prediction App")

# App description
st.write("Enter the customer details below to predict credit risk (Good or Bad).")

# Create input fields for all features
age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.selectbox("Job (0: unemployed, 1: unskilled, 2: skilled, 3: highly skilled)", options=[0, 1, 2, 3])
credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.number_input("Duration (in months)", min_value=1, max_value=72, value=12)
sex = st.selectbox("Sex", options=["male", "female"])
housing = st.selectbox("Housing", options=["own", "rent"])  # Removed 'free' as unseen
saving_accounts = st.selectbox("Saving Accounts", options=["moderate", "quite rich", "rich"])  # Removed 'little', 'no_info'
checking_account = st.selectbox("Checking Account", options=["moderate", "rich"])  # Removed 'little', 'no_info'
purpose = st.selectbox("Purpose", options=["car", "furniture/equipment", "radio/TV", "domestic appliances", 
                                          "repairs", "education"])  # Removed 'business', 'vacation/others'

# Button to trigger prediction
if st.button("Predict"):
    # Prepare input data
    input_data = {
        'Unnamed: 0': 0,  # Dummy value for index column
        'Age': age,
        'Job': job,
        'Credit amount': credit_amount,
        'Duration': duration,
        'Sex': sex,
        'Housing': housing,
        'Saving accounts': saving_accounts,
        'Checking account': checking_account,
        'Purpose': purpose
    }
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Perform one-hot encoding
    input_encoded = pd.get_dummies(input_df, columns=['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose'])
    
    # Hardcode expected columns (based on training with drop_first=True and Unnamed: 0)
    expected_columns = [
        'Unnamed: 0', 'Age', 'Job', 'Credit amount', 'Duration',
        'Sex_male', 'Housing_own', 'Housing_rent',
        'Saving accounts_moderate', 'Saving accounts_quite rich', 'Saving accounts_rich',
        'Checking account_moderate', 'Checking account_rich',
        'Purpose_car', 'Purpose_furniture/equipment', 'Purpose_radio/TV',
        'Purpose_domestic appliances', 'Purpose_repairs', 'Purpose_education'
    ]
    
    # Align input columns
    for col in expected_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[expected_columns]
    
    # Make prediction
    prediction = model.predict(input_encoded)
    
    # Display result
    result = "Good" if prediction[0] == "good" else "Bad"
    st.success(f"Predicted Credit Risk: **{result}**")

# Instructions
st.markdown("""
### Instructions
1. Enter the customer's details in the fields above.
2. Click the **Predict** button to see the credit risk prediction.
3. The model predicts whether the credit risk is **Good** (low risk) or **Bad** (high risk).
""")
