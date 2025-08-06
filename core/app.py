import streamlit as st
import pandas as pd

#page configuration
st.set_page_config(
    page_title="ADVANTEC WEB APP",
    page_icon="🛣️",
    layout="wide"
)

# 2. Main Title
st.title("Active Transportation & Operations Management Dashboard")

# 3. Centered, theme-adaptive dashboard objective/subheader
dashboard_objective = """
<div style="
    font-size: 1.15rem;
    font-weight: 400;
    color: var(--text-color);
    background: var(--background-color);
    padding: 1.2rem 1.5rem;
    border-radius: 14px;
    box-shadow: 0 2px 16px 0 var(--shadow-color, rgba(0,0,0,0.06));
    margin-bottom: 2rem;
    line-height: 1.7;
    ">
    <b>The ADVANTEC App</b> provides traffic engineering recommendations for the Coachella Valley using <b>MILLIONS OF DATA POINTS trained on Machine Learning Algorithms to REDUCE Travel Time, Fuel Consumption, and Green House Gases.</b> This is accomplished through the identification of anomalies, provision of cycle length recommendations, and predictive modeling.
</div>
"""

st.markdown(dashboard_objective, unsafe_allow_html=True)



# Load Data
df = pd.read_csv("https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/MOCK_DATA/mock_corridor_data.csv")


st.header("🛣️ Corridor Analysis")

# Sidebar with collapsible sections
with st.sidebar:
    st.title("🛣️ Controls")

    # Filter section
    with st.expander("📊 Data Filters", expanded=True):
        corridor = st.selectbox("Corridor", ["Option 1", "Option 2"])
        direction = st.radio("Direction", ["Northbound", "Southbound", "Both"])


    # Analysis tools
    with st.expander("🔧 Analysis Tools"):
        show_anomalies = st.checkbox("Show Anomalies")
        show_predictions = st.checkbox("Show Predictions")
        confidence_level = st.slider("Confidence Level", 80, 99, 95)

