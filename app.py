import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go


predict_button = st.sidebar.button("Run Diagnostics")

# Then indent your prediction code under this:
if predict_button:
    # Feature Engineering and Prediction code goes here...
# 1. Page Configuration
 st.set_page_config(page_title="Grinding Machine AI Monitor", page_icon="⚙️", layout="wide")

# 2. Load the Model
@st.cache_resource # This keeps the model loaded in memory so it's super fast
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Change line 15 to this:
    MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# 3. Sidebar: Control Room (Exact Number Inputs)
st.sidebar.title("⚙️ Live Sensor Data")
st.sidebar.markdown("Enter precise machine parameters below:")

# Using number_input to prevent the slider frustration
machine_type = st.sidebar.selectbox("Machine Quality Type", ["L", "M", "H"])
air_temp = st.sidebar.number_input("Air Temperature [K]", min_value=200.0, max_value=400.0, value=298.0, step=0.1)
process_temp = st.sidebar.number_input("Process Temperature [K]", min_value=200.0, max_value=400.0, value=308.0, step=0.1)
rpm = st.sidebar.number_input("Rotational Speed [rpm]", min_value=0, max_value=5000, value=1500, step=10)
torque = st.sidebar.number_input("Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0, step=0.5)
tool_wear = st.sidebar.number_input("Tool Wear [min]", min_value=0, max_value=500, value=100, step=1)

# Credits Section
st.sidebar.markdown("---")
st.sidebar.markdown("**Engineered by Parth Landge**")
st.sidebar.caption("Machine Learning Portfolio Project")

# 4. Main Dashboard Area
# 4. Main Dashboard Area
st.title("🏭 Predictive Maintenance Dashboard")
st.write("Real-time AI monitoring for grinding machine health and failure prevention.")

# The Button (Added use_container_width to make it look nicer!)
predict_button = st.sidebar.button("Run Diagnostics", use_container_width=True)

# 5. Process the inputs ONLY if the button is clicked
if predict_button:
    if model:
        # Feature Engineering
        temp_diff = process_temp - air_temp
        power = rpm * torque * (2 * np.pi / 60)
        type_L = 1 if machine_type == "L" else 0
        type_M = 1 if machine_type == "M" else 0

        input_df = pd.DataFrame([{
            'Air temperature [K]': air_temp,
            'Process temperature [K]': process_temp,
            'Rotational speed [rpm]': rpm,
            'Torque [Nm]': torque,
            'Tool wear [min]': tool_wear,
            'Temperature_Difference': temp_diff,
            'Power [W]': power,
            'Type_L': type_L,
            'Type_M': type_M
        }])

        # Predict using raw values to match the training array format
        probability = model.predict_proba(input_df.values)[0][1]
        prediction = int(model.predict(input_df.values)[0])

        # UI: The Health Gauge
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Create a beautiful Speedometer gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                number = {'suffix': "%"},
                title = {'text': "Failure Probability Risk"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "salmon"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("System Status")
            if prediction == 1:
                st.error("⚠️ **CRITICAL WARNING:** Machine Failure is imminent based on current operational parameters.")
                st.warning("*(Phase 3: The Generative AI RAG troubleshooting manual will appear here shortly...)*")
            else:
                st.success("✅ **STATUS: NORMAL.** Machine is operating within safe limits.")
                st.info("No immediate maintenance required. Keep monitoring.")

# What to show before the button is clicked
else:
    st.info("👈 Set your machine parameters in the sidebar and click **Run Diagnostics** to generate a health report.")
