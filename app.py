import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="FraudGuard AI - Insurance Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# Load trained model bundle
@st.cache_resource
def load_payload():
    with open('insurance_fraud_model.pkl', 'rb') as f:
        return pickle.load(f)

payload = load_payload()
pipeline = payload['pipeline']
metrics = payload['metrics']
eda_info = payload['eda_summary']

st.title("🛡️ FraudGuard AI: Vehicle Insurance Fraud Portal")
st.markdown("Assess claim fraud risks, inspect model evaluation metrics, explore dataset summary statistics, and view author information.")

tabs = st.tabs(["📋 Claim Fraud Prediction", "📊 Model Performance", "📈 Data Insights", "👤 About Us"])

# TAB 1: Complete Prediction Form (All 23 Input Features)
with tabs[0]:
    st.header("Assess New Claim")
    st.write("Fill in the 23 claim features below to predict the likelihood of fraud:")

    with st.form("fraud_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Driver & Policy Info")
            age_of_driver = st.number_input("Age of Driver", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["M", "F"])
            marital_status = st.selectbox("Marital Status", [1, 0], format_func=lambda x: "Married" if x == 1 else "Single / Other")
            safety_rating = st.slider("Safety Rating", 0, 100, 75)
            annual_income = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0, step=1000.0)
            high_education = st.selectbox("High Education Completed", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            address_change = st.selectbox("Address Change Recently", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            property_status = st.selectbox("Property Status", ["Own", "Rent"])

        with col2:
            st.subheader("Accident & Claim Details")
            accident_site = st.selectbox("Accident Site", ["Highway", "Local", "Parking Lot"])
            past_num_of_claims = st.number_input("Past Number of Claims", min_value=0, max_value=20, value=0)
            witness_present = st.selectbox("Witness Present", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            liab_prct = st.slider("Liability Percentage (%)", 0, 100, 25)
            channel = st.selectbox("Policy Channel", ["Phone", "Online", "Broker"])
            police_report = st.selectbox("Police Report Filed", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            days_open = st.number_input("Days Claim Open", min_value=0.0, value=8.0, step=0.5)
            form_defects = st.number_input("Form Defects Count", min_value=0, max_value=15, value=0)

        with col3:
            st.subheader("Vehicle & Financial Values")
            age_of_vehicle = st.number_input("Age of Vehicle (years)", min_value=0, max_value=30, value=4)
            vehicle_category = st.selectbox("Vehicle Category", ["Compact", "Medium", "Large"])
            vehicle_price = st.number_input("Vehicle Price ($)", min_value=0.0, value=25000.0, step=500.0)
            total_claim = st.number_input("Total Claim Amount ($)", min_value=0.0, value=12000.0, step=500.0)
            injury_claim = st.number_input("Injury Claim Amount ($)", min_value=0.0, value=3000.0, step=500.0)
            policy_deductible = st.selectbox("Policy Deductible ($)", [500, 1000, 2000])
            annual_premium = st.number_input("Annual Premium ($)", min_value=0.0, value=1200.0, step=100.0)

        submit_btn = st.form_submit_button("Evaluate Claim Fraud Risk", use_container_width=True)

    if submit_btn:
        input_data = pd.DataFrame([{
            'age_of_driver': age_of_driver,
            'marital_status': marital_status,
            'safety_rating': safety_rating,
            'annual_income': annual_income,
            'high_education': high_education,
            'address_change': address_change,
            'past_num_of_claims': past_num_of_claims,
            'witness_present': witness_present,
            'liab_prct': liab_prct,
            'police_report': police_report,
            'age_of_vehicle': age_of_vehicle,
            'vehicle_price': vehicle_price,
            'total_claim': total_claim,
            'injury_claim': injury_claim,
            'policy_deductible': policy_deductible,
            'annual_premium': annual_premium,
            'days_open': days_open,
            'form_defects': form_defects,
            'gender': gender,
            'property_status': property_status,
            'accident_site': accident_site,
            'channel': channel,
            'vehicle_category': vehicle_category
        }])

        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ **High Fraud Risk Detected** | Confidence Score: **{probability*100:.1f}%**")
        else:
            st.success(f"✅ **Low Fraud Risk / Legitimate Claim** | Confidence Score: **{(1-probability)*100:.1f}%**")

# TAB 2: Model Performance Metrics
with tabs[1]:
    st.header("GradientBoostingClassifier Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", metrics['accuracy'])
    m2.metric("F1 Score", metrics['f1_score'])
    m3.metric("ROC AUC", metrics['roc_auc'])

# TAB 3: Dataset Summary
with tabs[2]:
    st.header("Exploratory Data Analysis Summary")
    d1, d2, d3 = st.columns(3)
    d1.metric("Raw Claims Loaded", eda_info['raw_records'])
    d2.metric("Invalid Records Removed", eda_info['rows_removed'])
    d3.metric("Final Training Claims", eda_info['final_records'])

# TAB 4: About Us Section
with tabs[3]:
    st.header("About Us")

    st.markdown("""
    ### 👨‍💻 Project Developer Information

    **Developer Name: Jogi Gautam  
    **Role:** Machine Learning Engineer / Data Scientist  
    **Email: [codewithfun000@gmail.com]   

    ---

    ### 🚀 Project Overview
    **FraudGuard AI** is an end-to-end Machine Learning web application built to assist insurance companies in identifying fraudulent vehicle insurance claims.

    * **Machine Learning Pipeline:** Implements Scikit-Learn data transformation, handling dirty symbols, scaling, and hyperparameter-tuned classification (`GradientBoostingClassifier`).
    * **Key Capabilities:** Instant claim risk assessment, interactive feature parameter inputs, and real-time confidence scores.
    """)
