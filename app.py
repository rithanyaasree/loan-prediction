import streamlit as st
import pandas as pd
import joblib

# Load model and expected columns
model = joblib.load("loan_approval_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("Loan Approval Predictor")

with st.form("loan_form"):
    st.markdown("### Personal & Loan Information")
    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD", "Other"])

        with col2:
            employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-employed", "Student"])  
            purpose = st.selectbox("Purpose of Loan", ["Home", "Car", "Education", "Business", "Other"])

    st.markdown("### Financial Details")
    with st.container(border=True):
        annual_income = st.number_input("Annual Income ($)", min_value=0, step=1000)
        loan_amount = st.number_input("Loan Amount Requested ($)", min_value=0, step=1000)
        credit_score = st.slider("Credit Score", 300, 850, 700)

    submitted = st.form_submit_button("Check Loan Approval")

if submitted:
    try:
        input_dict = {
            "Gender": gender,
            "MaritalStatus": marital_status,
            "EducationLevel": education_level,
            "EmploymentStatus": employment_status,
            "PurposeOfLoan": purpose,
            "AnnualIncome": annual_income,
            "LoanAmountRequested": loan_amount,
            "CreditScore": credit_score
        }

        input_df = pd.DataFrame([input_dict])
        input_df = pd.get_dummies(input_df)

       
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[model_columns]  
        prediction = model.predict(input_df)[0]
        result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Denied"
        st.success(result)

    except Exception as e:
        st.error(f"Prediction Error: {e}")


