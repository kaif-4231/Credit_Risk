import streamlit as st
import pandas as pd
import joblib

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .sidebar .sidebar-content {
        background-color: #e8ecef;
        padding: 15px;
        border-radius: 10px;
    }
    .stSelectbox, .stNumberInput {
        background-color: white;
        border-radius: 5px;
        padding: 5px;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    h3 {
        color: #34495e;
        font-family: 'Arial', sans-serif;
    }
    .prediction-box {
        padding: 15px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
    .approved {
        background-color: #d4edda;
        color: #155724;
    }
    .denied {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and feature names
model = joblib.load("knn_model.pkl")

# Load the feature names used during training
try:
    expected_columns = joblib.load("model_features.pkl")
except FileNotFoundError:
    st.error("Feature list file (model_features.pkl) is missing. Ensure it is available in the same directory as this app.")
    st.stop()

# Function to collect user input
def user_input_features():
    st.sidebar.header("📋 Applicant Details")
    st.sidebar.write("Enter the details below to assess credit risk:")

    # Input fields in sidebar
    age = st.sidebar.number_input("Age (e.g., 35)", min_value=18, max_value=100, value=30, step=1, help="Enter the applicant's age")
    sex = st.sidebar.selectbox("Sex", ["male", "female"], help="Select the applicant's gender")
    job = st.sidebar.number_input("Job type (e.g., 0, 1, 2, 3)", min_value=0, max_value=3, value=1, step=1, help="Enter the job type (0: unskilled, 3: highly skilled)")
    housing = st.sidebar.selectbox("Housing", ["own", "free", "rent"], help="Select the applicant's housing status")
    saving_accounts = st.sidebar.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich", "unknown"], help="Select the saving account status")
    checking_account = st.sidebar.selectbox("Checking Account", ["little", "moderate", "rich", "unknown"], help="Select the checking account status")
    credit_amount = st.sidebar.number_input("Credit Amount (e.g., 1000)", min_value=0, value=1000, step=100, help="Enter the credit amount requested (in dollars)")
    duration = st.sidebar.number_input("Duration (in months, e.g., 12)", min_value=1, value=12, step=1, help="Enter the duration of the loan in months")
    purpose = st.sidebar.selectbox(
        "Purpose",
        ["business", "car", "domestic appliances", "education", "furniture/equipment", 
         "radio/TV", "repairs", "vacation/others"],
        help="Select the purpose of the loan"
    )

    # Collect inputs into a dictionary
    data = {
        "Age": age,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving_accounts,
        "Checking account": checking_account,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": purpose
    }

    return pd.DataFrame([data])

# Preprocess the user input to match the trained model
def preprocess_input(input_df):
    # Match preprocessing from training (e.g., one-hot encoding, scaling)
    categorical_columns = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
    
    # One-hot encode categorical columns
    one_hot_encoded = pd.get_dummies(input_df[categorical_columns], drop_first=False)
    
    # Drop original categorical columns and add one-hot-encoded columns
    input_df = input_df.drop(categorical_columns, axis=1).join(one_hot_encoded)
    
    # Add missing columns with zeros if necessary
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Ensure column order matches the expected order
    input_df = input_df[expected_columns]
    
    return input_df

# Main Streamlit App
def main():
    st.title("🏦 Credit Risk Prediction")
    st.write("This application predicts whether a loan application will be **Approved** or **Denied** based on the applicant's details.")
    st.markdown("---")

    # Display a welcome message or placeholder in the main area
    st.subheader("Welcome to the Credit Risk Predictor")
    st.write("Use the sidebar on the left to enter the applicant's details and click 'Predict Risk' to see the result.")

    input_df = user_input_features()

    if st.button("Predict Risk"):
        # Preprocess user input
        processed_input = preprocess_input(input_df)

        # Debugging: Show processed input
        with st.expander("🔍 Processed Input (Debugging)"):
            st.write("The processed input data used for prediction:")
            st.dataframe(processed_input)

        try:
            # Make prediction
            prediction = model.predict(processed_input)
            prediction_label = "Approved" if prediction[0] == 0 else "Denied"

            # Display prediction with styled box
            if prediction_label == "Approved":
                st.markdown(f'<div class="prediction-box approved">Prediction: {prediction_label}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box denied">Prediction: {prediction_label}</div>', unsafe_allow_html=True)
        except ValueError as e:
            st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
