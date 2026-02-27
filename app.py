import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# Custom Styling
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    /* Title styling */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        letter-spacing: 1px;
    }

    .sub-title {
        font-size: 1.1rem;
        color: #a0aec0;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }

    .result-safe {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.1));
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .result-fraud {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.1));
        border: 1px solid rgba(239, 68, 68, 0.4);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 0 20px 10px rgba(239, 68, 68, 0.1); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
    }

    .result-icon {
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }

    .result-text {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .result-desc {
        font-size: 1rem;
        color: #a0aec0;
    }

    /* Info card */
    .info-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(5px);
    }

    .info-card h4 {
        color: #63b3ed;
        margin-bottom: 0.5rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    /* Feature input section */
    .feature-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #63b3ed;
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid rgba(99, 179, 237, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("decision_tree_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()


# ─────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🛡️ Credit Card Fraud Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by Decision Tree Classifier — Enter transaction details to detect fraud in real-time</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Sidebar — Model Info
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Model Information")

    st.markdown("""
    <div class="info-card">
        <h4>🤖 Model</h4>
        <p>Decision Tree Classifier</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>📈 Performance</h4>
        <p>Accuracy: <b>99.92%</b></p>
        <p>Precision: <b>77.62%</b></p>
        <p>Recall: <b>74.50%</b></p>
        <p>F1-Score: <b>76.03%</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>📋 Dataset</h4>
        <p>283,726 transactions</p>
        <p>Fraud: 473 (0.17%)</p>
        <p>Genuine: 283,253 (99.83%)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Input Mode")
    input_mode = st.radio(
        "Choose how to enter features:",
        ["Manual Input", "Upload CSV"],
        label_visibility="collapsed",
    )


# ─────────────────────────────────────────────────────────
# Feature Names
# ─────────────────────────────────────────────────────────
feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


# ─────────────────────────────────────────────────────────
# Manual Input Mode
# ─────────────────────────────────────────────────────────
if input_mode == "Manual Input":

    st.markdown('<div class="feature-header">💳 Transaction Details</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        time_val = st.number_input(
            "⏱️ Time (seconds since first transaction)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            help="Seconds elapsed between this transaction and the first transaction in the dataset.",
        )
    with col2:
        amount_val = st.number_input(
            "💰 Transaction Amount ($)",
            min_value=0.0,
            value=0.0,
            step=0.01,
            help="The monetary amount of the transaction.",
        )

    st.markdown('<div class="feature-header">🔬 PCA-Transformed Features (V1 — V28)</div>', unsafe_allow_html=True)
    st.caption("These are anonymized features obtained via PCA transformation. Enter values if available, or leave as 0.")

    v_features = {}
    cols = st.columns(4)
    for i in range(1, 29):
        col_idx = (i - 1) % 4
        with cols[col_idx]:
            v_features[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0,
                step=0.01,
                format="%.4f",
                key=f"v{i}",
            )

    st.markdown("---")

    if st.button("🔍  Detect Fraud", use_container_width=True, type="primary"):
        # Build feature vector
        features = [time_val]
        for i in range(1, 29):
            features.append(v_features[f"V{i}"])
        features.append(amount_val)

        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)

        st.markdown("---")

        if prediction[0] == 1:
            st.markdown("""
            <div class="result-card result-fraud">
                <div class="result-icon">🚨</div>
                <div class="result-text" style="color: #ef4444;">FRAUD DETECTED</div>
                <div class="result-desc">This transaction has been flagged as potentially fraudulent.</div>
            </div>
            """, unsafe_allow_html=True)
            st.error("⚠️ Recommended Action: Block this transaction and alert the cardholder immediately.")
        else:
            st.markdown("""
            <div class="result-card result-safe">
                <div class="result-icon">✅</div>
                <div class="result-text" style="color: #10b981;">LEGITIMATE TRANSACTION</div>
                <div class="result-desc">This transaction appears to be genuine.</div>
            </div>
            """, unsafe_allow_html=True)
            st.success("✅ This transaction looks safe. No action required.")


# ─────────────────────────────────────────────────────────
# CSV Upload Mode
# ─────────────────────────────────────────────────────────
else:
    st.markdown('<div class="feature-header">📁 Upload Transaction Data</div>', unsafe_allow_html=True)
    st.caption("Upload a CSV file with columns: Time, V1–V28, Amount")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Validate columns
            missing = [col for col in feature_names if col not in df.columns]
            if missing:
                st.error(f"❌ Missing columns: {', '.join(missing)}")
            else:
                st.markdown("### 📋 Uploaded Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

                if st.button("🔍  Detect Fraud for All Transactions", use_container_width=True, type="primary"):
                    input_data = df[feature_names].values
                    predictions = model.predict(input_data)

                    # Add predictions to dataframe
                    result_df = df.copy()
                    result_df["Prediction"] = predictions
                    result_df["Status"] = result_df["Prediction"].map(
                        {0: "✅ Legitimate", 1: "🚨 Fraud"}
                    )

                    # Summary stats
                    total = len(predictions)
                    fraud_count = int(predictions.sum())
                    legit_count = total - fraud_count

                    st.markdown("---")
                    st.markdown("### 📊 Results Summary")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Transactions", total)
                    c2.metric("Legitimate", legit_count)
                    c3.metric("Fraudulent", fraud_count)

                    if fraud_count > 0:
                        st.warning(f"⚠️ {fraud_count} potentially fraudulent transaction(s) detected!")
                    else:
                        st.success("✅ All transactions appear legitimate.")

                    st.markdown("### 📋 Detailed Results")
                    st.dataframe(
                        result_df[["Time", "Amount", "Status"]],
                        use_container_width=True,
                        height=400,
                    )

                    # Download results
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="fraud_detection_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")


# ─────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #4a5568; font-size: 0.85rem;">'
    "Credit Card Fraud Detection System • Built with Streamlit & scikit-learn"
    "</p>",
    unsafe_allow_html=True,
)