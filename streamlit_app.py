import streamlit as st
import pickle
import pandas as pd
import time

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
        color: white;
    }
    .subtitle {
        font-size: 18px;
        color: #9aa0a6;
        margin-bottom: 30px;
    }
    .card {
        background-color: #111827;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.6);
        margin-bottom: 20px;
    }
    .risk-high {
        color: #ef4444;
        font-weight: 600;
        font-size: 20px;
    }
    .risk-low {
        color: #22c55e;
        font-weight: 600;
        font-size: 20px;
    }
    div[data-testid="stContainer"] {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    with open("churn_pipeline.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction Dashboard</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Analyze customer behavior and predict churn risk using machine learning.</div>',
    unsafe_allow_html=True
)

# Sidebar Inputs
st.sidebar.header("Customer Information")

tenure = st.sidebar.slider(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    value=12
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges ($)",
    min_value=0.0,
    value=70.0
)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer",
        "Credit card"
    ]
)

predict_btn = st.sidebar.button("🔍 Predict Churn Risk")

# Layout
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📥 Input Summary")

    st.write(f"**Tenure:** {tenure} months")
    st.write(f"**Monthly Charges:** ${monthly_charges:.2f}")
    st.write(f"**Contract Type:** {contract}")
    st.write(f"**Payment Method:** {payment_method}")

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Prediction Result")

    if predict_btn:
        input_data = pd.DataFrame([{
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "Contract": contract,
            "PaymentMethod": payment_method
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100

        # Dynamic decimal counter
        placeholder = st.empty()
        step = 0.1  # smaller step for smooth decimal increments
        current = 0.0
        while current < probability:
            placeholder.progress(int(current))  # progress bar needs int
            placeholder.metric("Estimated Churn Probability", f"{current:.2f}%")
            current += step
            time.sleep(0.0001)  

        # Ensure final value matches exactly
        placeholder.progress(int(probability))
        placeholder.metric("Estimated Churn Probability", f"{probability:.2f}%")

        # Show risk message
        if prediction == 1:
            st.markdown(
                '<div class="risk-high">⚠️ High Churn Risk</div>',
                unsafe_allow_html=True
            )
            st.write(
                "This customer shows patterns commonly associated with churn. "
                "Consider proactive retention strategies."
            )
        else:
            st.markdown(
                '<div class="risk-low">✅ Low Churn Risk</div>',
                unsafe_allow_html=True
            )
            st.write(
                "This customer is likely to remain loyal based on current behavior."
            )
    else:
        st.info(
            "Enter customer details and click **Predict Churn Risk** to see results."
        )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "Machine Learning Powered Customer Churn Prediction | Developed by Chamod Lakshitha"
)

