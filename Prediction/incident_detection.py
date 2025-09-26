# Prediction/incident_detection.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Prediction.timeline_scrubber import render_gradient_header, create_timeline_scrubber, create_hourly_forecast_table


def generate_incident_data():
    """
    Generate synthetic incident detection data for demonstration.
    In production, this would connect to real traffic sensors and ML models.
    """
    # Sample incident data
    current_time = datetime.now().replace(minute=0, second=0, microsecond=0)

    # Incident detected at 3PM
    incident_time = current_time.replace(hour=15)

    # Generate hourly data from 12PM to 12AM
    hours = list(range(12, 24)) + [0]  # 12PM to 12AM (midnight)

    incident_data = {
        'hour': hours,
        'travel_time_predicted': [4.1, 5.3, 8.8, 6.3, 4.4, 4.7, 4.2, 4.0, 3.8, 3.9, 4.1, 4.0, 3.9],
        'travel_time_actual': [4.2, 5.3, 8.1, 6.4, 4.4, 4.7, 4.1, 4.0, 3.9, 3.8, 4.2, 4.1, 3.8],
        'incident_status': ['🟢', '🟢', '🔴', '🟡', '🟡', '🟢', '🟢', '🟢', '🟢', '🟢', '🟢', '🟢', '🟢'],
        'conditions': ['Normal', 'Normal', 'INCIDENT', 'Recovery', 'Recovery', 'Normal', 'Normal', 'Normal', 'Normal',
                       'Normal', 'Normal', 'Normal', 'Normal']
    }

    return incident_data, incident_time


def create_incident_metrics():
    """Create KPI metrics for incident detection."""
    return {
        'time_to_clear': {'value': 9.1, 'unit': 'min', 'status': 'CLEARING'},
        'travel_time': {'value': 6.4, 'unit': 'min', 'baseline': 4.2},
        'incident_confidence': {'value': 91, 'unit': '%'},
        'model_accuracy': {'value': 94.2, 'unit': '%'},
        'detection_accuracy': {'value': 96, 'unit': '%'},
        'recovery_prediction': {'value': 100, 'unit': '%'}
    }


def render_incident_status_section(incident_data, incident_time):
    """Render the incident detection status section."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        border: 2px solid #ff4757; border-radius: 12px; padding: 16px; margin: 12px 0;
        font-family: 'Consolas', 'Monaco', monospace; color: white;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.5rem;">⚠️</span>
            <div>
                <div style="font-weight: 800; font-size: 1.1rem;">INCIDENT DETECTED & VALIDATED - September 19, 2024</div>
                <div style="font-size: 0.95rem; opacity: 0.9;">
                    Type: Sudden Spike • Recovery • Real-time Confidence: 91% • Post-Analysis: ✅ Confirmed
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_incident_kpis():
    """Render incident detection KPIs in ASCII-style layout."""
    metrics = create_incident_metrics()

    st.markdown("##### ### KPIs")

    # Create three columns for KPIs
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="
            background: rgba(255,107,107,0.1); border: 1px solid #ff6b6b;
            border-radius: 8px; padding: 12px; margin: 4px 0;
            font-family: 'Consolas', 'Monaco', monospace;">
            <div style="font-size: 0.8rem; color: #666;">⏱️ Time to Clear</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #ff6b6b;">9.1 min</div>
            <div style="font-size: 0.75rem; color: #999;">Peak→Baseline</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            background: rgba(52,152,219,0.1); border: 1px solid #3498db;
            border-radius: 8px; padding: 12px; margin: 4px 0;
            font-family: 'Consolas', 'Monaco', monospace;">
            <div style="font-size: 0.8rem; color: #666;">🚗 Travel Time</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #3498db;">6.4 min</div>
            <div style="font-size: 0.75rem; color: #999;">vs 4.2 baseline</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="
            background: rgba(46,204,113,0.1); border: 1px solid #2ecc71;
            border-radius: 8px; padding: 12px; margin: 4px 0;
            font-family: 'Consolas', 'Monaco', monospace;">
            <div style="font-size: 0.8rem; color: #666;">🚨 Incident Status</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #2ecc71;">CLEARING</div>
            <div style="font-size: 0.75rem; color: #999;">75% improvement</div>
        </div>
        """, unsafe_allow_html=True)


def render_prediction_accuracy_section():
    """Render prediction model accuracy metrics."""
    st.markdown("##### ### Prediction Model Accuracy")

    # Create accuracy metrics layout
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div style="
            background: rgba(155,89,182,0.1); border: 1px solid #9b59b6;
            border-radius: 8px; padding: 10px; text-align: center;
            font-family: 'Consolas', 'Monaco', monospace;">
            <div style="color: #9b59b6; font-weight: 800;">🎯 Detection</div>
            <div style="font-size: 1.2rem; font-weight: 800;">91%</div>
            <div style="font-size: 0.7rem; color: #999;">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            background: rgba(52,152,219,0.1); border: 1px solid #3498db;
            border-radius: 8px; padding: 10px; text-align: center;
            font-family: 'Consolas', 'Monaco', monospace;">
            <div style="color: #3498db; font-weight: 800;">📊 Peak Travel Time</div>
            <div style="font-size: 1.2rem; font-weight: 800;">96%</div>
            <div style="font-size: 0.7rem; color: #999;">Prediction</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="
            background: rgba(241,196,15,0.1); border: 1px solid #f1c40f;
            border-radius: 8px; padding: 10px; text-align: center;
            font-family: 'Consolas', 'Monaco', monospace;">
            <div style="color: #f39c12; font-weight: 800;">⏰ Time To Clear</div>
            <div style="font-size: 1.2rem; font-weight: 800;">100%</div>
            <div style="font-size: 0.7rem; color: #999;">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="
            background: rgba(46,204,113,0.1); border: 1px solid #2ecc71;
            border-radius: 8px; padding: 10px; text-align: center;
            font-family: 'Consolas', 'Monaco', monospace;">
            <div style="color: #2ecc71; font-weight: 800;">🔄 Model</div>
            <div style="font-size: 1.2rem; font-weight: 800;">PASS</div>
            <div style="font-size: 0.7rem; color: #999;">Validation</div>
        </div>
        """, unsafe_allow_html=True)


def render_incident_timeline_table(incident_data):
    """Render the incident timeline with predicted vs actual comparison."""
    st.markdown("##### ### Hourly Incident Timeline - Predicted vs Actual")

    # Create the timeline table
    hours = incident_data['hour']
    predicted = incident_data['travel_time_predicted']
    actual = incident_data['travel_time_actual']
    status = incident_data['incident_status']
    conditions = incident_data['conditions']

    # Hour headers
    hour_cols = st.columns(len(hours))
    for i, hour in enumerate(hours):
        if hour == 0:
            hour_display = "12AM"
        elif hour == 12:
            hour_display = "12PM"
        elif hour < 12:
            hour_display = f"{hour}PM"
        else:
            hour_display = f"{hour - 12}PM" if hour > 12 else f"{hour}PM"

        with hour_cols[i]:
            st.markdown(f"""
            <div style="
                text-align: center; padding: 8px; background: rgba(52,152,219,0.1);
                border: 1px solid #3498db; border-radius: 6px; margin: 2px 0;
                font-family: 'Consolas', 'Monaco', monospace; font-weight: 800;">
                [{hour_display}]
            </div>
            """, unsafe_allow_html=True)

    # Predicted row
    pred_cols = st.columns(len(predicted))
    for i, (pred, stat) in enumerate(zip(predicted, status)):
        with pred_cols[i]:
            st.markdown(f"""
            <div style="text-align: center; margin: 4px 0; font-family: 'Consolas', 'Monaco', monospace;">
                <div style="font-size: 1.1rem;">{stat}</div>
                <div style="font-weight: 800; color: #3498db;">{pred:.1f} min</div>
                <div style="font-size: 0.75rem; color: #999;">PREDICT</div>
            </div>
            """, unsafe_allow_html=True)

    # Actual row
    act_cols = st.columns(len(actual))
    for i, act in enumerate(actual):
        with act_cols[i]:
            st.markdown(f"""
            <div style="text-align: center; margin: 4px 0; font-family: 'Consolas', 'Monaco', monospace;">
                <div style="font-weight: 800; color: #27ae60;">{act:.1f} min</div>
                <div style="font-size: 0.75rem; color: #999;">ACTUAL</div>
            </div>
            """, unsafe_allow_html=True)

    # Error row
    err_cols = st.columns(len(predicted))
    for i, (pred, act) in enumerate(zip(predicted, actual)):
        error = abs(pred - act) / act * 100 if act > 0 else 0
        color = "#27ae60" if error <= 5 else ("#f39c12" if error <= 15 else "#e74c3c")

        with err_cols[i]:
            st.markdown(f"""
            <div style="text-align: center; margin: 2px 0; font-family: 'Consolas', 'Monaco', monospace;">
                <div style="font-weight: 800; color: {color};">±{error:.1f}%</div>
                <div style="font-size: 0.65rem; color: #999;">ERROR</div>
            </div>
            """, unsafe_allow_html=True)


def render_incident_characteristics():
    """Render incident characteristics section."""
    st.markdown("##### 🚨 INCIDENT CHARACTERISTICS:")

    characteristics = [
        "• Peak Impact: 5.1 min travel time (+17% vs 4.2 min baseline)",
        "• Within-Hour Volatility: 8.2 min range (Max 11.4 - Min 6.2)",
        "• Recovery Pattern: Hour 6 shows Classic Clearing (high start, low finish)",
        "• Return to Baseline: Hour 7 (4.4 min - within baseline range)"
    ]

    for char in characteristics:
        st.markdown(f"""
        <div style="
            font-family: 'Consolas', 'Monaco', monospace;
            color: #2c3e50; font-size: 0.9rem; margin: 4px 0;">
            {char}
        </div>
        """, unsafe_allow_html=True)


def render_incident_detection_section():
    """
    Main function to render the complete Incident Detection & Recovery section.
    This matches the ASCII layout design shown in the images.
    """
    # Load synthetic data
    incident_data, incident_time = generate_incident_data()

    # Render gradient header
    render_gradient_header(
        title="Incident Detection & Recovery Analysis: Washington Street NB",
        subtitle_left="Jun 29, 2025 to Jul 29, 2025 | Validated Pattern Analysis | Model Performance Verified",
        icon="⚠️"
    )

    # Goal statement
    st.markdown("""
    <div style="
        background: rgba(231,76,60,0.1); border-left: 4px solid #e74c3c;
        padding: 12px; margin: 12px 0; border-radius: 4px;">
        <span style="color: #e74c3c; font-weight: 800;">🎯 GOAL:</span> 
        <span style="color: #2c3e50;">Detect incidents when they happen and estimate how long until conditions return to a free-flow state</span>
    </div>
    """, unsafe_allow_html=True)

    # Timeline scrubber
    hour_list, selected_hour = create_timeline_scrubber(
        center_hour=15,  # 3PM incident time
        date_label="Sep 19, 2024",
        window_size=7,
        key_prefix="incident_timeline"
    )

    # Incident status section
    render_incident_status_section(incident_data, incident_time)

    # KPIs section
    render_incident_kpis()

    # Timeline analysis
    st.markdown("---")
    render_incident_timeline_table(incident_data)

    # Model accuracy section
    st.markdown("---")
    render_prediction_accuracy_section()

    # Incident characteristics
    st.markdown("---")
    render_incident_characteristics()

    # Additional analysis sections
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **PATTERN STRENGTH**

        87%  
        Very Strong

        🔥 High Conf
        """, help="Pattern recognition confidence for incident type classification")

    with col2:
        st.markdown("""
        **PREDICTION ACCURACY**

        94.2%  
        Excellent

        📊 Current 5-Hr Range
        """, help="Model accuracy over the next 5 hour prediction window")

    with col3:
        st.markdown("""
        **TIMELINE**

        Next 3 Hours  
        [===] 50%

        ⏰ ETA: 47min
        """, help="Expected time to full recovery based on historical patterns")

    # Footer with model performance
    st.markdown("---")
    st.markdown("""
    <div style="
        background: rgba(46,204,113,0.1); border: 1px solid #2ecc71;
        border-radius: 8px; padding: 12px; margin: 12px 0;
        font-family: 'Consolas', 'Monaco', monospace; text-align: center;">
        <span style="color: #2ecc71; font-weight: 800;">🟢 EXCELLENT</span>
        <span style="margin: 0 20px;">🟢 EXCELLENT</span>
        <span style="color: #2ecc71; font-weight: 800;">🟢 EXCELLENT</span>
        <span style="margin: 0 20px;">🟢 EXCELLENT</span>
    </div>
    """, unsafe_allow_html=True)

    # Performance summary
    st.info("""
    **Model Performance Summary:**
    - Detected incident correctly with 91% confidence
    - Only 0.3 min error in travel time prediction vs actual: 8.1 min 
    - Perfect prediction of recovery timeline (4.4 min → within baseline range)
    - Validated against 23/25 similar patterns from historical data
    - Model caught this incident: Average model vs Actual: 2 hours with 91.2% accuracy, 83.4% confidence
    """)