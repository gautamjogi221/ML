import streamlit as st
import pickle
import pandas as pd
import numpy as np

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_payload():
    with open("insurance_fraud_model.pkl", "rb") as f:
        return pickle.load(f)

payload = load_payload()

pipeline = payload["pipeline"]
metrics = payload["metrics"]
eda_info = payload["eda_summary"]

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* ==============================
       GLOBAL
    ============================== */

    .stApp {
        background:
            radial-gradient(circle at top right,
            rgba(0, 229, 255, 0.08),
            transparent 30%),
            #07111f;
        color: #E8F1F8;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1500px;
    }

    /* ==============================
       SIDEBAR
       ============================== */

    section[data-testid="stSidebar"] {
        background: #050D18;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] h1 {
        color: #00E5FF;
    }

    /* ==============================
       HEADERS
       ============================== */

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
        background: linear-gradient(
            90deg,
            #FFFFFF,
            #00E5FF
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #8FA6BA;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    /* ==============================
       CARDS
       ============================== */

    .glass-card {
        background: rgba(15, 29, 46, 0.85);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow:
            0 10px 30px rgba(0,0,0,0.20);
    }

    .metric-card {
        background: linear-gradient(
            145deg,
            #102337,
            #0A1727
        );
        border: 1px solid rgba(0,229,255,0.15);
        border-radius: 18px;
        padding: 22px;
        text-align: center;
    }

    .metric-title {
        color: #8FA6BA;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        color: #00E5FF;
        font-size: 30px;
        font-weight: 800;
        margin-top: 8px;
    }

    /* ==============================
       FORM
       ============================== */

    div[data-testid="stForm"] {
        background: rgba(10, 24, 40, 0.90);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 25px;
    }

    /* ==============================
       BUTTON
       ============================== */

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(
            90deg,
            #00B8D4,
            #00E5FF
        );
        color: #031018;
        border: none;
        border-radius: 12px;
        font-weight: 800;
        padding: 12px 25px;
        transition: 0.3s;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,229,255,0.25);
    }

    /* ==============================
       INPUTS
       ============================== */

    div[data-baseweb="input"],
    div[data-baseweb="select"] {
        border-radius: 10px;
    }

    /* ==============================
       RESULT CARDS
       ============================== */

    .risk-high {
        background: linear-gradient(
            135deg,
            rgba(255, 65, 65, 0.18),
            rgba(120, 20, 20, 0.25)
        );
        border: 1px solid rgba(255,65,65,0.35);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-top: 25px;
    }

    .risk-low {
        background: linear-gradient(
            135deg,
            rgba(0, 220, 150, 0.15),
            rgba(0, 100, 80, 0.25)
        );
        border: 1px solid rgba(0,220,150,0.30);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-top: 25px;
    }

    .risk-title {
        font-size: 28px;
        font-weight: 800;
    }

    .risk-score {
        font-size: 45px;
        font-weight: 900;
        margin-top: 10px;
    }

    /* ==============================
       ABOUT
       ============================== */

    .about-card {
        background: linear-gradient(
            145deg,
            #0E2034,
            #081523
        );
        border: 1px solid rgba(0,229,255,0.15);
        border-radius: 20px;
        padding: 30px;
    }

    .developer-name {
        font-size: 30px;
        font-weight: 800;
        color: #00E5FF;
    }

    .badge {
        display: inline-block;
        background: rgba(0,229,255,0.10);
        color: #00E5FF;
        border: 1px solid rgba(0,229,255,0.20);
        border-radius: 20px;
        padding: 6px 12px;
        margin: 4px;
        font-size: 13px;
    }

    /* ==============================
       FOOTER
       ============================== */

    .footer {
        text-align: center;
        color: #657C91;
        padding: 30px;
        margin-top: 50px;
        border-top: 1px solid rgba(255,255,255,0.05);
    }

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div style="text-align:center; padding:20px 0;">

        <div style="
            font-size:60px;
            margin-bottom:10px;
        ">
            🛡️
        </div>

        <h1 style="
            font-size:25px;
            margin:0;
        ">
            FraudGuard AI
        </h1>

        <p style="
            color:#71879A;
            font-size:13px;
        ">
            Insurance Fraud Intelligence
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🚀 Platform")

    st.markdown("""
    **AI Claim Analysis**

    Machine-learning powered insurance claim risk assessment.
    """)

    st.divider()

    st.markdown("### 🤖 Model")

    st.markdown("""
    **Algorithm**

    Gradient Boosting Classifier

    **Prediction**

    Binary Fraud Classification
    """)

    st.divider()

    st.caption("FraudGuard AI v1.0")
    st.caption("Machine Learning Project")

# =========================================================
# HERO HEADER
# =========================================================

st.markdown(
    '<div class="hero-title">🛡️ FraudGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'AI-powered vehicle insurance fraud detection and claim intelligence platform.'
    '</div>',
    unsafe_allow_html=True
)

# =========================================================
# TOP STATUS CARDS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Model</div>
        <div class="metric-value">GB</div>
        <div style="color:#71879A;">Gradient Boosting</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Accuracy</div>
        <div class="metric-value">{metrics['accuracy']}</div>
        <div style="color:#71879A;">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">F1 Score</div>
        <div class="metric-value">{metrics['f1_score']}</div>
        <div style="color:#71879A;">Classification Balance</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">ROC AUC</div>
        <div class="metric-value">{metrics['roc_auc']}</div>
        <div style="color:#71879A;">Model Separation</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =========================================================
# NAVIGATION TABS
# =========================================================

tabs = st.tabs([
    "🔍 Claim Analysis",
    "📊 Model Performance",
    "📈 Data Insights",
    "👨‍💻 About Project"
])

# =========================================================
# TAB 1 - CLAIM PREDICTION
# =========================================================

with tabs[0]:

    st.markdown(
        '<div class="section-title">🔍 New Claim Risk Assessment</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            color:#8FA6BA;
            margin-bottom:20px;
        ">
        Enter the claim information below. The AI model will analyze
        23 claim characteristics and estimate the fraud risk.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("fraud_form"):

        # ---------------------------------------------
        # DRIVER
        # ---------------------------------------------

        st.markdown("### 👤 Driver & Policy Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            age_of_driver = st.number_input(
                "Age of Driver",
                min_value=18,
                max_value=100,
                value=35
            )

            gender = st.selectbox(
                "Gender",
                ["M", "F"]
            )

            marital_status = st.selectbox(
                "Marital Status",
                [1, 0],
                format_func=lambda x:
                    "Married" if x == 1 else "Single / Other"
            )

        with col2:

            safety_rating = st.slider(
                "Safety Rating",
                0,
                100,
                75
            )

            annual_income = st.number_input(
                "Annual Income ($)",
                min_value=0.0,
                value=50000.0,
                step=1000.0
            )

            high_education = st.selectbox(
                "Higher Education Completed",
                [1, 0],
                format_func=lambda x:
                    "Yes" if x == 1 else "No"
            )

        with col3:

            address_change = st.selectbox(
                "Recent Address Change",
                [0, 1],
                format_func=lambda x:
                    "No" if x == 0 else "Yes"
            )

            property_status = st.selectbox(
                "Property Status",
                ["Own", "Rent"]
            )

        st.divider()

        # ---------------------------------------------
        # ACCIDENT
        # ---------------------------------------------

        st.markdown("### 🚗 Accident & Claim Details")

        col1, col2, col3 = st.columns(3)

        with col1:

            accident_site = st.selectbox(
                "Accident Site",
                [
                    "Highway",
                    "Local",
                    "Parking Lot"
                ]
            )

            past_num_of_claims = st.number_input(
                "Previous Claims",
                min_value=0,
                max_value=20,
                value=0
            )

            witness_present = st.selectbox(
                "Witness Present",
                [0, 1],
                format_func=lambda x:
                    "Yes" if x == 1 else "No"
            )

        with col2:

            liab_prct = st.slider(
                "Liability Percentage",
                0,
                100,
                25
            )

            channel = st.selectbox(
                "Policy Channel",
                [
                    "Phone",
                    "Online",
                    "Broker"
                ]
            )

            police_report = st.selectbox(
                "Police Report Filed",
                [0, 1],
                format_func=lambda x:
                    "Yes" if x == 1 else "No"
            )

        with col3:

            days_open = st.number_input(
                "Days Claim Open",
                min_value=0.0,
                value=8.0,
                step=0.5
            )

            form_defects = st.number_input(
                "Form Defects Count",
                min_value=0,
                max_value=15,
                value=0
            )

        st.divider()

        # ---------------------------------------------
        # VEHICLE
        # ---------------------------------------------

        st.markdown("### 🚘 Vehicle & Financial Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            age_of_vehicle = st.number_input(
                "Vehicle Age (Years)",
                min_value=0,
                max_value=30,
                value=4
            )

            vehicle_category = st.selectbox(
                "Vehicle Category",
                [
                    "Compact",
                    "Medium",
                    "Large"
                ]
            )

            vehicle_price = st.number_input(
                "Vehicle Price ($)",
                min_value=0.0,
                value=25000.0,
                step=500.0
            )

        with col2:

            total_claim = st.number_input(
                "Total Claim Amount ($)",
                min_value=0.0,
                value=12000.0,
                step=500.0
            )

            injury_claim = st.number_input(
                "Injury Claim Amount ($)",
                min_value=0.0,
                value=3000.0,
                step=500.0
            )

        with col3:

            policy_deductible = st.selectbox(
                "Policy Deductible ($)",
                [500, 1000, 2000]
            )

            annual_premium = st.number_input(
                "Annual Premium ($)",
                min_value=0.0,
                value=1200.0,
                step=100.0
            )

        st.write("")

        submit_btn = st.form_submit_button(
            "🛡️  ANALYZE CLAIM RISK",
            use_container_width=True
        )

    # =====================================================
    # PREDICTION
    # =====================================================

    if submit_btn:

        input_data = pd.DataFrame([{

            "age_of_driver": age_of_driver,
            "marital_status": marital_status,
            "safety_rating": safety_rating,
            "annual_income": annual_income,
            "high_education": high_education,
            "address_change": address_change,
            "past_num_of_claims": past_num_of_claims,
            "witness_present": witness_present,
            "liab_prct": liab_prct,
            "police_report": police_report,
            "age_of_vehicle": age_of_vehicle,
            "vehicle_price": vehicle_price,
            "total_claim": total_claim,
            "injury_claim": injury_claim,
            "policy_deductible": policy_deductible,
            "annual_premium": annual_premium,
            "days_open": days_open,
            "form_defects": form_defects,
            "gender": gender,
            "property_status": property_status,
            "accident_site": accident_site,
            "channel": channel,
            "vehicle_category": vehicle_category

        }])

        prediction = pipeline.predict(input_data)[0]

        probability = pipeline.predict_proba(input_data)[0][1]

        st.markdown("## 🎯 AI Assessment")

        if prediction == 1:

            st.markdown(
                f"""
                <div class="risk-high">

                    <div style="font-size:55px;">⚠️</div>

                    <div class="risk-title">
                        HIGH FRAUD RISK
                    </div>

                    <div style="
                        color:#B8C7D3;
                        margin-top:8px;
                    ">
                        The submitted claim has been classified
                        as potentially fraudulent.
                    </div>

                    <div class="risk-score">
                        {probability * 100:.1f}%
                    </div>

                    <div style="color:#9AAEBE;">
                        Fraud Probability
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="risk-low">

                    <div style="font-size:55px;">✅</div>

                    <div class="risk-title">
                        LOW FRAUD RISK
                    </div>

                    <div style="
                        color:#B8C7D3;
                        margin-top:8px;
                    ">
                        The submitted claim has been classified
                        as low risk by the AI model.
                    </div>

                    <div class="risk-score">
                        {(1 - probability) * 100:.1f}%
                    </div>

                    <div style="color:#9AAEBE;">
                        Legitimate Claim Confidence
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

# =========================================================
# TAB 2 - MODEL PERFORMANCE
# =========================================================

with tabs[1]:

    st.markdown(
        '<div class="section-title">📊 Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="glass-card">

    <h3>Gradient Boosting Classifier</h3>

    <p style="color:#8FA6BA;">
    The FraudGuard AI prediction engine uses a Gradient Boosting
    classification model to identify potentially fraudulent
    vehicle insurance claims.
    </p>

    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(f"""
        <div class="metric-card">

            <div class="metric-title">
                Accuracy
            </div>

            <div class="metric-value">
                {metrics['accuracy']}
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown(f"""
        <div class="metric-card">

            <div class="metric-title">
                F1 Score
            </div>

            <div class="metric-value">
                {metrics['f1_score']}
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown(f"""
        <div class="metric-card">

            <div class="metric-title">
                ROC AUC
            </div>

            <div class="metric-value">
                {metrics['roc_auc']}
            </div>

        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 3 - DATA INSIGHTS
# =========================================================

with tabs[2]:

    st.markdown(
        '<div class="section-title">📈 Dataset Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="glass-card">

        <h3>Exploratory Data Analysis</h3>

        <p style="color:#8FA6BA;">
        Overview of the dataset used during model development
        and preprocessing.
        </p>

    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(f"""
        <div class="metric-card">

            <div class="metric-title">
                Raw Claims
            </div>

            <div class="metric-value">
                {eda_info['raw_records']:,}
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown(f"""
        <div class="metric-card">

            <div class="metric-title">
                Records Removed
            </div>

            <div class="metric-value">
                {eda_info['rows_removed']:,}
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown(f"""
        <div class="metric-card">

            <div class="metric-title">
                Final Claims
            </div>

            <div class="metric-value">
                {eda_info['final_records']:,}
            </div>

        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Dataset progress

    final_records = eda_info["final_records"]
    raw_records = eda_info["raw_records"]

    if raw_records > 0:

        retention = final_records / raw_records

        st.markdown("### 📊 Dataset Retention")

        st.progress(
            min(retention, 1.0)
        )

        st.caption(
            f"{retention * 100:.1f}% of the original records "
            f"were retained for model development."
        )

# =========================================================
# TAB 4 - ABOUT
# =========================================================

with tabs[3]:

    st.markdown(
        '<div class="section-title">👨‍💻 About FraudGuard AI</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="about-card">

        <div style="font-size:55px;">
            👨‍💻
        </div>

        <div class="developer-name">
            Jogi Gautam
        </div>

        <p style="
            color:#8FA6BA;
            font-size:17px;
        ">
            Machine Learning Engineer / Data Scientist
        </p>

        <p>
            📧 <b>codewithfun000@gmail.com</b>
        </p>

        <div style="margin-top:20px;">

            <span class="badge">
                Python
            </span>

            <span class="badge">
                Machine Learning
            </span>

            <span class="badge">
                Scikit-Learn
            </span>

            <span class="badge">
                Streamlit
            </span>

            <span class="badge">
                Data Science
            </span>

        </div>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="glass-card">

        <h2>🚀 Project Overview</h2>

        <p style="color:#A7B7C5; line-height:1.8;">

        <b>FraudGuard AI</b> is an end-to-end Machine Learning
        web application designed to assist in identifying
        potentially fraudulent vehicle insurance claims.

        </p>

        <hr style="border-color:rgba(255,255,255,0.08);">

        <h3>🧠 Machine Learning Pipeline</h3>

        <p style="color:#A7B7C5;">
        The application uses a Scikit-Learn machine-learning
        pipeline for data preprocessing, transformation and
        Gradient Boosting classification.
        </p>

        <h3>⚡ Key Capabilities</h3>

        <ul style="color:#A7B7C5; line-height:2;">

            <li>Real-time insurance claim risk prediction</li>

            <li>23-feature claim analysis</li>

            <li>Fraud probability estimation</li>

            <li>Model performance monitoring</li>

            <li>Dataset insights</li>

            <li>Interactive Streamlit dashboard</li>

        </ul>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

    🛡️ <b>FraudGuard AI</b>
    <br>
    AI-Powered Insurance Fraud Detection
    <br><br>
    <span style="font-size:12px;">
        Machine Learning Project • Built with Python & Streamlit
    </span>

</div>
""", unsafe_allow_html=True)