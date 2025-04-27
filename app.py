import streamlit as st
import pandas as pd
import joblib

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
    st.subheader("Applicant Details")
    st.write("Enter the details below to assess credit risk:")

    col1, col2 = st.columns(2)

    # Input fields
    with col1:
        age = st.number_input("Age (e.g., 35)", min_value=18, max_value=100, value=30, step=1, help="Enter the applicant's age")
        job = st.number_input("Job type (e.g., 0, 1, 2, 3)", min_value=0, max_value=3, value=1, step=1, help="Enter the job type (0: unskilled, 3: highly skilled)")
        housing = st.selectbox("Housing", ["own", "free", "rent"], help="Select the applicant's housing status")
        saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich", "unknown"], help="Select the saving account status")
        credit_amount = st.number_input("Credit Amount (e.g., 1000)", min_value=0, value=1000, step=100, help="Enter the credit amount requested (in dollars)")

    with col2:
        sex = st.selectbox("Sex", ["male", "female"], help="Select the applicant's gender")
        checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich", "unknown"], help="Select the checking account status")
        duration = st.number_input("Duration (in months, e.g., 12)", min_value=1, value=12, step=1, help="Enter the duration of the loan in months")
        purpose = st.selectbox(
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

    input_df = user_input_features()

    if st.button("Predict Risk"):
        # Preprocess user input
        processed_input = preprocess_input(input_df)

        # Debugging: Show processed input
        with st.expander("Processed Input (Debugging)"):
            st.write("The processed input data used for prediction:")
            st.dataframe(processed_input)

        try:
            # Make prediction
            prediction = model.predict(processed_input)
            prediction_label = "Approved" if prediction[0] == 0 else "Denied"

            # Display prediction with color coding
            if prediction_label == "Approved":
                st.success(f"### Prediction: {prediction_label}")
            else:
                st.error(f"### Prediction: {prediction_label}")
        except ValueError as e:
            st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
