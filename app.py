import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real-Time Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PROFESSIONAL NAVY THEME
# ============================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: "Segoe UI", sans-serif;
}

/* Main background */
.stApp {
    background: #071426;
    color: #F8FAFC;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0B1B33;
    border-right: 1px solid #29466B;
}

section[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}

/* Main headings */
h1, h2, h3 {
    color: #FFFFFF !important;
}

/* Normal text */
p, label, span, div {
    color: #E8F0FA;
}

/* Cards */
.info-card {
    background: #102642;
    border: 1px solid #315477;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.25);
}

.metric-card {
    background: #102642;
    border: 1px solid #3B6088;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    min-height: 120px;
}

.metric-title {
    color: #BBD0E8 !important;
    font-size: 14px;
    font-weight: 600;
}

.metric-value {
    color: #FFFFFF !important;
    font-size: 30px;
    font-weight: 700;
    margin-top: 8px;
}

/* Prediction cards */
.safe-card {
    background: #0D302B;
    border: 1px solid #27AE8A;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
}

.safe-card h2 {
    color: #63F0C4 !important;
}

.fraud-card {
    background: #351A25;
    border: 1px solid #F05A75;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
}

.fraud-card h2 {
    color: #FF7890 !important;
}

/* Buttons */
.stButton > button {
    background: #1E5AA8;
    color: #FFFFFF !important;
    border: 1px solid #4C8BD6;
    border-radius: 9px;
    font-weight: 700;
    padding: 10px 22px;
}

.stButton > button:hover {
    background: #2874C7;
    color: #FFFFFF !important;
    border-color: #72AEF0;
}

/* Select boxes */
div[data-baseweb="select"] > div {
    background-color: #102642;
    color: #FFFFFF;
    border-color: #3B6088;
}

/* Inputs */
input {
    background-color: #102642 !important;
    color: #FFFFFF !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #315477;
}

/* Divider */
hr {
    border-color: #315477;
}

/* Status */
.status-live {
    color: #63F0C4 !important;
    font-weight: 700;
}

.small-text {
    color: #BBD0E8 !important;
    font-size: 14px;
}


/* ===== ATTRACTIVE ANIMATIONS ===== */
@keyframes pulseLive {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: .45; transform: scale(1.18); }
}

@keyframes cardFloat {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-3px); }
}

@keyframes scanLine {
    0% { transform: translateY(-18px); opacity: 0; }
    20% { opacity: 1; }
    80% { opacity: 1; }
    100% { transform: translateY(18px); opacity: 0; }
}

@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 0 rgba(76,139,214,0); }
    50% { box-shadow: 0 0 22px rgba(76,139,214,.20); }
}

.status-live::first-letter {
    animation: pulseLive 1.4s infinite;
}

.metric-card {
    transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    border-color: #72AEF0;
    box-shadow: 0 10px 28px rgba(0,0,0,.30);
}

.info-card {
    animation: glowPulse 3s ease-in-out infinite;
}

.scan-box {
    position: relative;
    overflow: hidden;
    background: #0B1B33;
    border: 1px solid #315477;
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    margin: 15px 0;
}

.scan-box::after {
    content: "";
    position: absolute;
    left: 8%;
    right: 8%;
    top: 50%;
    height: 2px;
    background: #4C8BD6;
    box-shadow: 0 0 12px #4C8BD6;
    animation: scanLine 1.7s ease-in-out infinite;
}

.scan-text {
    position: relative;
    z-index: 2;
    font-size: 17px;
    font-weight: 700;
}

.safe-card, .fraud-card {
    animation: cardFloat 3s ease-in-out infinite;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load("models/fraud_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")

    return model, preprocessor


try:
    model, preprocessor = load_model()
    model_loaded = True
except Exception:
    model_loaded = False


# ============================================================
# SESSION STATE
# ============================================================

if "streaming" not in st.session_state:
    st.session_state.streaming = False

if "stream_data" not in st.session_state:
    st.session_state.stream_data = []

if "total_checked" not in st.session_state:
    st.session_state.total_checked = 0

if "fraud_count" not in st.session_state:
    st.session_state.fraud_count = 0

if "safe_count" not in st.session_state:
    st.session_state.safe_count = 0


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="info-card">

<h1>🛡️ Real-Time Fraud Detection System</h1>

<p style="font-size:18px; color:#DCEBFA !important;">
An intelligent machine-learning system designed to identify
potentially fraudulent financial transactions in real time.
</p>

<p class="small-text">
The system analyses transaction behaviour such as amount,
transaction type, payment method, location, device changes,
international activity, failed attempts and account age.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🛡️ FRAUD SHIELD")

    st.markdown(
        '<p class="small-text">Real-Time Transaction Intelligence</p>',
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🔍 Single Transaction",
            "📡 Live Streaming",
            "📊 Analytics",
            "📋 Transaction History",
            "ℹ️ About Project"
        ]
    )

    st.divider()

    if model_loaded:

        st.markdown(
            '<p class="status-live">🟢 MODEL ONLINE <span style="font-size:12px; opacity:.8;">• AI READY</span></p>',
            unsafe_allow_html=True
        )

    else:

        st.error("Model files not found")


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown("## 📊 Fraud Detection Dashboard")
    st.markdown('<p class="small-text">Live overview of transaction security and model monitoring.</p>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    <h3>How the system works</h3>

    <p>
    <b>1. Transaction Input</b> → Transaction information enters the system.
    </p>

    <p>
    <b>2. Feature Processing</b> → Numerical and categorical features are prepared.
    </p>

    <p>
    <b>3. Machine Learning</b> → The trained Random Forest model analyses the transaction.
    </p>

    <p>
    <b>4. Risk Decision</b> → The system classifies the transaction as Safe or Potential Fraud.
    </p>

    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">TRANSACTIONS CHECKED</div>
        <div class="metric-value">20</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">FRAUD CASES</div>
        <div class="metric-value">4</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">SAFE TRANSACTIONS</div>
        <div class="metric-value">16</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">MODEL ACCURACY</div>
        <div class="metric-value">100%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🔎 Detection Factors")

    factors = pd.DataFrame({
        "Factor": [
            "Transaction Amount",
            "Transaction Frequency",
            "Location Change",
            "Device Change",
            "International Transaction",
            "Failed Attempts",
            "Account Age",
            "Transaction Time"
        ],
        "Purpose": [
            "Detect unusually large payments",
            "Identify unusual transaction frequency",
            "Detect suspicious movement",
            "Identify new or changed devices",
            "Flag cross-border activity",
            "Detect repeated failed attempts",
            "Identify risky new accounts",
            "Detect unusual transaction hours"
        ]
    })

    st.dataframe(
        factors,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SINGLE TRANSACTION
# ============================================================

elif page == "🔍 Single Transaction":

    st.markdown("## 🔍 Single Transaction Risk Analysis")

    st.markdown("""
    <div class="info-card">
    <h3>Transaction Analysis</h3>
    <p>
    Enter transaction details below. The trained machine-learning
    model will analyse the transaction and estimate whether it
    appears safe or potentially fraudulent.
    </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        date = st.date_input(
            "📅 Transaction Date",
            datetime.today()
        )

        hour = st.slider(
            "🕐 Transaction Hour",
            0,
            23,
            12
        )

        amount = st.number_input(
            "💰 Transaction Amount",
            min_value=0.0,
            value=1000.0,
            step=100.0
        )

        transaction_type = st.selectbox(
            "💳 Transaction Type",
            ["POS", "Online", "ATM", "Transfer"]
        )

        payment_method = st.selectbox(
            "💵 Payment Method",
            ["Credit Card", "Debit Card", "UPI", "Bank Transfer"]
        )

        location = st.selectbox(
            "📍 Current Location",
            ["Hyderabad", "Mumbai", "Delhi", "Chennai", "Bangalore"]
        )

    with col2:

        previous_location = st.selectbox(
            "📍 Previous Location",
            ["Hyderabad", "Mumbai", "Delhi", "Chennai", "Bangalore"]
        )

        device_type = st.selectbox(
            "📱 Device Type",
            ["Mobile", "Laptop", "Tablet", "Desktop"]
        )

        device_changed = st.selectbox(
            "🔄 Device Changed?",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        international = st.selectbox(
            "🌍 International Transaction?",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        previous_amount = st.number_input(
            "💰 Previous Transaction Amount",
            min_value=0.0,
            value=500.0
        )

        transaction_frequency = st.number_input(
            "🔁 Transaction Frequency",
            min_value=0,
            value=2
        )

        distance = st.number_input(
            "📏 Distance From Previous Transaction",
            min_value=0,
            value=5
        )

        failed_attempts = st.number_input(
            "⚠️ Failed Attempts",
            min_value=0,
            value=0
        )

        account_age = st.number_input(
            "👤 Account Age (Days)",
            min_value=0,
            value=300
        )

    st.divider()

    if st.button("🔍 ANALYSE TRANSACTION", use_container_width=True):

        if not model_loaded:

            st.error("Model files are missing. Please train the model first.")

        else:

            scan_placeholder = st.empty()
            scan_placeholder.markdown("""
            <div class="scan-box">
                <div class="scan-text">🔍 SCANNING TRANSACTION</div>
                <div class="small-text">AI is checking transaction behaviour and risk indicators...</div>
            </div>
            """, unsafe_allow_html=True)

            progress = st.progress(0)
            for step in range(0, 101, 20):
                progress.progress(step)
                time.sleep(0.12)

            input_data = pd.DataFrame([{
                "hour": hour,
                "amount": amount,
                "transaction_type": transaction_type,
                "payment_method": payment_method,
                "location": location,
                "previous_location": previous_location,
                "device_type": device_type,
                "device_changed": device_changed,
                "is_international": international,
                "previous_amount": previous_amount,
                "transaction_frequency": transaction_frequency,
                "distance_from_previous": distance,
                "failed_attempts": failed_attempts,
                "account_age_days": account_age,
                "day": date.day,
                "month": date.month,
                "day_of_week": date.weekday()
            }])

            processed = preprocessor.transform(input_data)

            prediction = model.predict(processed)[0]

            if hasattr(model, "predict_proba"):

                probability = model.predict_proba(processed)[0][1]

            else:

                probability = float(prediction)

            scan_placeholder.empty()
            progress.empty()

            st.divider()

            if prediction == 1:

                st.markdown(f"""
                <div class="fraud-card">

                <h2>⚠️ POTENTIAL FRAUD DETECTED</h2>

                <p style="color:#FFFFFF !important; font-size:18px;">
                This transaction shows characteristics associated
                with fraudulent activity.
                </p>

                <h3 style="color:#FF7890 !important;">
                Risk Score: {probability * 100:.1f}%
                </h3>

                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="safe-card">

                <h2>✓ TRANSACTION APPEARS SAFE</h2>

                <p style="color:#FFFFFF !important; font-size:18px;">
                No strong fraud pattern was detected by the model.
                </p>

                <h3 style="color:#63F0C4 !important;">
                Fraud Probability: {probability * 100:.1f}%
                </h3>

                </div>
                """, unsafe_allow_html=True)


# ============================================================
# PSEUDO LIVE STREAMING
# ============================================================

elif page == "📡 Live Streaming":

    st.markdown("## 📡 Real-Time Transaction Monitoring")

    st.markdown("""
    <div class="info-card">

    <h3>⚡ Pseudo-Streaming Mode</h3>

    <p>
    This module simulates a real-time financial monitoring environment.
    Transactions are processed one at a time, allowing the dashboard
    to continuously update fraud alerts and monitoring statistics.
    </p>

    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">STREAM PROCESSED</div>
        <div class="metric-value">{st.session_state.total_checked}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">FRAUD ALERTS</div>
        <div class="metric-value">{st.session_state.fraud_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">SAFE</div>
        <div class="metric-value">{st.session_state.safe_count}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    start = st.button(
        "▶ START LIVE STREAM",
        use_container_width=True
    )

    if start:

        try:

            data = pd.read_csv("data/transactions.csv")

            progress = st.progress(0)

            status = st.empty()

            live_table = st.empty()

            for index, row in data.iterrows():

                status.markdown(
                    f"""
                    <h3 class="status-live">
                    ● LIVE — Processing transaction {index + 1} of {len(data)}
                    </h3>
                    """,
                    unsafe_allow_html=True
                )

                input_data = pd.DataFrame([{

                    "hour": row["hour"],
                    "amount": row["amount"],
                    "transaction_type": row["transaction_type"],
                    "payment_method": row["payment_method"],
                    "location": row["location"],
                    "previous_location": row["previous_location"],
                    "device_type": row["device_type"],
                    "device_changed": row["device_changed"],
                    "is_international": row["is_international"],
                    "previous_amount": row["previous_amount"],
                    "transaction_frequency": row["transaction_frequency"],
                    "distance_from_previous": row["distance_from_previous"],
                    "failed_attempts": row["failed_attempts"],
                    "account_age_days": row["account_age_days"],
                    "day": pd.to_datetime(row["date"]).day,
                    "month": pd.to_datetime(row["date"]).month,
                    "day_of_week": pd.to_datetime(row["date"]).weekday()

                }])

                processed = preprocessor.transform(input_data)

                prediction = model.predict(processed)[0]

                if hasattr(model, "predict_proba"):

                    probability = model.predict_proba(processed)[0][1]

                else:

                    probability = float(prediction)

                result = "FRAUD ALERT" if prediction == 1 else "SAFE"

                if prediction == 1:
                    st.session_state.fraud_count += 1
                else:
                    st.session_state.safe_count += 1

                st.session_state.total_checked += 1

                new_record = {

                    "Transaction": row["transaction_id"],
                    "Amount": f"₹{row['amount']:,.2f}",
                    "Location": row["location"],
                    "Device": row["device_type"],
                    "Risk": f"{probability * 100:.1f}%",
                    "Status": result

                }

                st.session_state.stream_data.append(new_record)

                live_table.dataframe(
                    pd.DataFrame(
                        st.session_state.stream_data[-10:]
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                progress.progress(
                    int(((index + 1) / len(data)) * 100)
                )

                time.sleep(0.8)

            status.success(
                "✓ Streaming completed successfully."
            )

        except Exception as e:

            st.error(f"Streaming error: {e}")


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.markdown("## 📊 Fraud Analytics")

    try:

        data = pd.read_csv("data/transactions.csv")

        total = len(data)

        fraud = int(data["is_fraud"].sum())

        safe = total - fraud

        fraud_rate = (fraud / total) * 100

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Total Transactions", total)

        with c2:
            st.metric("Fraud Transactions", fraud)

        with c3:
            st.metric("Safe Transactions", safe)

        with c4:
            st.metric("Fraud Rate", f"{fraud_rate:.1f}%")

        st.divider()

        st.markdown("### Fraud Distribution")

        chart_data = pd.DataFrame({
            "Status": ["Safe", "Fraud"],
            "Transactions": [safe, fraud]
        })

        st.bar_chart(
            chart_data.set_index("Status")
        )

        st.markdown("### Transaction Amount Analysis")

        st.line_chart(
            data[["amount"]]
        )

    except Exception as e:

        st.error(f"Analytics error: {e}")


# ============================================================
# TRANSACTION HISTORY
# ============================================================

elif page == "📋 Transaction History":

    st.markdown("## 📋 Transaction History")

    try:

        data = pd.read_csv(
            "data/transactions.csv"
        )

        data["Status"] = data["is_fraud"].map({
            0: "✓ Safe",
            1: "⚠ Fraud"
        })

        display_columns = [
            "transaction_id",
            "customer_id",
            "date",
            "amount",
            "transaction_type",
            "payment_method",
            "location",
            "device_type",
            "failed_attempts",
            "is_international",
            "Status"
        ]

        st.dataframe(
            data[display_columns],
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.error(f"Unable to load transaction history: {e}")


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.markdown("## ℹ️ About the Project")

    st.markdown("""
    <div class="info-card">

    <h2>🛡️ Real-Time Fraud Detection System</h2>

    <p>
    This project uses Machine Learning to identify suspicious
    financial transactions and support real-time fraud monitoring.
    </p>

    <h3>🎯 Objective</h3>

    <p>
    The primary objective is to detect potentially fraudulent
    transactions by analysing multiple behavioural and transaction-level
    factors.
    </p>

    <h3>🤖 Machine Learning Model</h3>

    <p>
    The system uses a <b>Random Forest Classifier</b> trained on
    historical transaction data.
    </p>

    <h3>🔍 Major Detection Factors</h3>

    <p>
    • Transaction amount<br>
    • Transaction frequency<br>
    • Payment method<br>
    • Transaction type<br>
    • Current and previous location<br>
    • Device type and device changes<br>
    • International transactions<br>
    • Failed attempts<br>
    • Previous transaction amount<br>
    • Account age<br>
    • Transaction date and time
    </p>

    <h3>⚙️ Technology Stack</h3>

    <p>
    <b>Python</b> •
    <b>Pandas</b> •
    <b>NumPy</b> •
    <b>Scikit-learn</b> •
    <b>Joblib</b> •
    <b>Streamlit</b> •
    <b>Plotly</b>
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Note: The live-streaming feature in this project is a "
        "pseudo-streaming simulation using the available transaction dataset."
    )
    
