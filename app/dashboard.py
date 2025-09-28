import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import pickle
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- UI Configuration & Styling ---
st.set_page_config(
    page_title="Cognitive Shield AI SOC",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

# Inject custom CSS for a more polished look
st.markdown("""
    <style>
    .stMetric {
        border-radius: 10px;
        padding: 15px;
        background-color: #262730;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #ff4b4b, #ff7f7f);
    }
    </style>
""", unsafe_allow_html=True)


# --- Configuration & Model Paths ---
EVENTS_FILE = '../data/events.csv'
USERS_FILE = '../data/users.csv'
MODEL_C_PATH = '../src/models/engine_c_lgbm.pkl'

# --- Main Loading Function (Cached for performance) ---
@st.cache_resource
def load_all_artifacts():
    """Loads all models, data, and pre-computed artifacts with robust error checking."""
    if not all(os.path.exists(f) for f in [EVENTS_FILE, USERS_FILE, MODEL_C_PATH]):
        st.error("One or more essential files (events.csv, users.csv, engine_c_lgbm.pkl) are missing. Please run the notebooks first.", icon="🚨")
        st.stop()
        
    df = pd.read_csv(EVENTS_FILE, parse_dates=['timestamp'])
    users_df = pd.read_csv(USERS_FILE, parse_dates=['customer_since'])
    
    user_stats = df.groupby('user_id')['amount'].agg(['mean', 'std', 'count', 'max']).reset_index()
    user_stats.columns = ['user_id', 'avg_amount', 'std_amount', 'tx_count', 'max_amount']
    user_stats = user_stats.fillna(0)

    with open(MODEL_C_PATH, 'rb') as f:
        model_c = pickle.load(f)
    explainer_c = shap.TreeExplainer(model_c)
    
    return df, users_df, user_stats, model_c, explainer_c

# --- Main App ---
st.title("🧠 Cognitive Shield AI: Security Operations Center")
st.caption(f"Jaipur, Rajasthan | System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')} | Status: ✅ Online")

# --- Load Artifacts ---
with st.spinner("Initializing AI engines and loading terabytes of historical data..."):
    df, users_df, user_stats, model_c, explainer_c = load_all_artifacts()

# --- Sidebar Controls ---
st.sidebar.header("🕹️ Simulation Controls")
st.sidebar.markdown("Generate a new transaction event to be analyzed by the Cognitive Shield system.")

analysis_type = st.sidebar.radio(
    "Select Transaction Type:",
    ('Random Legitimate', 'Account Takeover Fraud', 'Mule Ring Fraud'),
    help="Choose a specific scenario to see how the AI engines react."
)

if st.sidebar.button("Process New Transaction", type="primary", use_container_width=True):
    if analysis_type == 'Account Takeover Fraud':
        # Find a session with a password reset followed by a high-value transfer
        pw_reset_sessions = df[df['event_type'] == 'password_reset']['session_id']
        sample_tx = df[(df['session_id'].isin(pw_reset_sessions)) & (df['event_type'] == 'high_value_transfer')].sample(1).iloc[0]
    elif analysis_type == 'Mule Ring Fraud':
        # Find a p2p transfer from a user with only a few transactions (likely a mule)
        mule_candidates = user_stats[user_stats['tx_count'] < 5]['user_id']
        sample_tx = df[(df['is_fraud'] == True) & (df['user_id'].isin(mule_candidates)) & (df['event_type'] == 'p2p_transfer')].sample(1).iloc[0]
    else: # Legitimate
        sample_tx = df[df['is_fraud'] == False].sample(1).iloc[0]
    st.session_state['sample_tx'] = sample_tx
    st.session_state['last_analysis'] = analysis_type

# --- Main Dashboard Display ---
if 'sample_tx' not in st.session_state:
    st.info("👈 Please click 'Process New Transaction' in the sidebar to begin analysis.")
    st.image("https://i.imgur.com/gY92cAG.png", caption="Cognitive Shield AI Architecture")
else:
    tx = st.session_state['sample_tx']
    user_id = tx['user_id']
    is_fraud = tx['is_fraud']
    
    # --- SCORING LOGIC ---
    # Mock scores for Engine A & B
    score_a = np.random.uniform(0.7, 0.95) if is_fraud else np.random.uniform(0.01, 0.25)
    score_b = np.random.uniform(0.8, 0.99) if is_fraud else np.random.uniform(0.0, 0.2)
    
    # Live scoring for Engine C
    tx_df = pd.DataFrame([tx])
    tx_df = tx_df.merge(user_stats, on='user_id', how='left').fillna(0)
    tx_df['amount_deviation'] = (tx_df['amount'] - tx_df['avg_amount']) / (tx_df['std_amount'] + 1e-6)
    tx_df['hour_of_day'] = tx_df['timestamp'].dt.hour
    tx_df['day_of_week'] = tx_df['timestamp'].dt.dayofweek
    tx_df['is_weekend'] = (tx_df['day_of_week'] >= 5).astype(int)
    tx_df['is_business_hours'] = ((tx_df['hour_of_day'] >= 9) & (tx_df['hour_of_day'] <= 18)).astype(int)
    tx_df['time_since_last_event_sec'] = np.random.randint(60, 3600)
    tx_df['is_new_device_for_user'] = 1 if np.random.rand() > (0.1 if is_fraud else 0.9) else 0
    tx_df['events_in_session'] = len(df[df['session_id'] == tx['session_id']])
    features_for_model = ['amount', 'avg_amount', 'std_amount', 'tx_count', 'amount_deviation', 'hour_of_day', 'day_of_week', 'is_weekend', 'is_business_hours', 'time_since_last_event_sec', 'is_new_device_for_user', 'events_in_session']
    score_c = model_c.predict_proba(tx_df[features_for_model])[:, 1][0]
    final_score = (0.35 * score_a) + (0.35 * score_b) + (0.30 * score_c)

    # --- HEADER & VERDICT ---
    st.header(f"Incident Analysis: `{tx['event_id']}`", divider='rainbow')
    
    verdict_col1, verdict_col2 = st.columns([1, 2])
    with verdict_col1:
        st.subheader("Final Verdict")
        if final_score > 0.6:
            st.error(f"**High Risk ({final_score:.2f})**", icon="🚨")
            st.markdown("#### Recommended Action: **Block & Escalate**")
        elif final_score > 0.3:
            st.warning(f"**Medium Risk ({final_score:.2f})**", icon="⚠️")
            st.markdown("#### Recommended Action: **Step-Up Auth**")
        else:
            st.success(f"**Low Risk ({final_score:.2f})**", icon="✅")
            st.markdown("#### Recommended Action: **Allow**")
            
    with verdict_col2:
        st.subheader("Incident Summary")
        st.markdown(f"""
        - **Timestamp:** `{tx['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}`
        - **User:** `{user_id}`
        - **Action:** `{tx['event_type'].replace('_', ' ').title()}` for **₹{tx['amount']:,.2f}**
        - **Scenario Demoed:** `{st.session_state['last_analysis']}`
        """)

    # --- TABS FOR DETAILED ANALYSIS ---
    tab1, tab2, tab3 = st.tabs(["**🔬 AI Engine Analysis**", "**👤 Contextual Intelligence**", "**📖 Full History**"])

    with tab1: # AI Engine Analysis
        st.header("Deep Dive into AI Engine Scores")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Engine A: Behavioral DNA", help="Analyzes the sequence of actions in a session.")
            st.metric("Anomaly Score", f"{score_a:.2f}")
            st.progress(score_a)
            if score_a > 0.5:
                st.warning("Sequence of actions is highly unusual for this user.", icon="⚠️")
            else:
                st.success("User behavior is consistent with past patterns.", icon="✅")

        with c2:
            st.subheader("Engine B: Relationship Cortex", help="Analyzes connections between users, devices, and beneficiaries.")
            st.metric("Network Risk Score", f"{score_b:.2f}")
            st.progress(score_b)
            if score_b > 0.5:
                st.warning("Connections to high-risk entities detected.", icon="🔗")
            else:
                st.success("No suspicious links in the transaction graph.", icon="✅")
        
        with c3:
            st.subheader("Engine C: Causal Engine", help="Analyzes the context and 'common sense' of the transaction.")
            st.metric("Contextual Risk Score", f"{score_c:.2f}")
            st.progress(score_c)
            if score_c > 0.5:
                st.warning("Action lacks a normal causal trigger.", icon="❓")
            else:
                st.success("Transaction context appears logical.", icon="✅")
        
        st.divider()
        st.subheader("Explainable AI: Top Risk Factors (from Engine C)")
        shap_values = explainer_c.shap_values(tx_df[features_for_model])
        shap_values_for_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
        base_value = explainer_c.expected_value[1] if isinstance(explainer_c.expected_value, list) else explainer_c.expected_value
        
        # --- FIX: Removed the problematic plt.tight_layout() call ---
        # The SHAP plot manages its own layout, so tight_layout() is not needed and can cause rendering conflicts.
        fig, ax = plt.subplots(figsize=(10, 2.5))
        shap.force_plot(base_value, shap_values_for_plot[0,:], tx_df[features_for_model].iloc[0,:], matplotlib=True, show=False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig) # Explicitly close the plot
        
        st.caption("Red features push the risk score higher (towards fraud), Blue features push it lower.")

    with tab2: # Contextual Intelligence
        st.header("Investigative Context")
        uc1, uc2 = st.columns(2)
        with uc1:
            st.subheader("👤 User Intelligence")
            if user_id in users_df['user_id'].values:
                user_profile = users_df[users_df['user_id'] == user_id].iloc[0]
                user_stat_row = user_stats[user_stats['user_id'] == user_id].iloc[0]
                st.write(f"**Customer Since:** {user_profile['customer_since'].strftime('%Y-%m-%d')}")
                st.write(f"**KYC Risk Level:** `{user_profile['kyc_risk_level'].upper()}`")
                st.write(f"**Avg/Max Transaction:** ₹{user_stat_row['avg_amount']:,.2f} / ₹{user_stat_row['max_amount']:,.2f}")
            else:
                st.error("User profile not found. This is a major red flag.", icon="❗️")

        with uc2:
            st.subheader("💻 Session Intelligence")
            st.write(f"**Device ID:** `{tx['device_id']}`")
            st.write(f"**IP Address:** `{tx['ip_address']}` (Geolocation: Jaipur, IN - Mocked)")
            st.write(f"**Channel:** `{tx['channel']}`")
        
        st.subheader("Historical Activity (Last 30 Days)")
        user_history = df[(df['user_id'] == user_id) & (df['timestamp'] > df['timestamp'].max() - timedelta(days=30))]
        if not user_history.empty:
            daily_activity = user_history.set_index('timestamp').resample('D')['amount'].sum()
            
            # --- ROBUST PLOTTING FIX & VISUAL UPGRADE ---
            # Use a dark theme for the plot to match our dashboard
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 4))

            # Plot the data with improved aesthetics
            ax.plot(daily_activity.index, daily_activity.values, marker='o', linestyle='-', color='#ff4b4b', label='Daily Total')
            ax.fill_between(daily_activity.index, daily_activity.values, color='#ff4b4b', alpha=0.3)

            # Improve styling for a professional SOC look
            ax.set_title(f"User {user_id[-6:]}'s Daily Transaction Volume (Last 30 Days)", color='white', fontsize=14)
            ax.set_ylabel("Amount (INR)", color='white')
            ax.tick_params(axis='x', colors='white', rotation=45)
            ax.tick_params(axis='y', colors='white')
            ax.grid(True, linestyle='--', alpha=0.3)
            fig.patch.set_facecolor('#0e1117') # Match Streamlit's dark background
            ax.set_facecolor('#262730') # Match metric background
            plt.tight_layout()

            st.pyplot(fig, use_container_width=True)
            plt.close(fig) # Explicitly close the figure to free up memory
        else:
            st.write("No historical activity found for this user in the last 30 days.")

    with tab3: # Full History
        st.header(f"Full Transaction History for User: `{user_id}`")
        st.dataframe(df[df['user_id'] == user_id].sort_values('timestamp', ascending=False), use_container_width=True, hide_index=True)



