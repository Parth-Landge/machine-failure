import streamlit as st
import plotly.graph_objects as go
import requests  # <-- New library to send data over the network

# 1. Page Configuration
st.set_page_config(page_title="Predictive Maintenance", layout="wide")

# URL of your FastAPI backend engine (running via Uvicorn locally)
API_URL = "https://milling-machine-api.onrender.com/predict"

# 2. Sidebar Controls (The Inputs)
st.sidebar.header("🛠️ Machine Settings")
machine_type = st.sidebar.selectbox("Machine Quality Type", ["L", "M", "H"])
air_temp = st.sidebar.number_input("Air Temperature [K]", value=298.0)
process_temp = st.sidebar.number_input("Process Temperature [K]", value=308.0)
rpm = st.sidebar.number_input("Rotational Speed [rpm]", value=1500.0)
torque = st.sidebar.number_input("Torque [Nm]", value=40.0)
tool_wear = st.sidebar.number_input("Tool Wear [min]", value=50.0)

# 3. Main Dashboard Area
st.title("🏭 Decoupled Predictive Maintenance System")
st.write("Frontend UI connected directly to the central FastAPI Engine.")

predict_button = st.sidebar.button("Run Live Diagnostics", use_container_width=True)

if predict_button:
    # Package the data exactly how our FastAPI SensorData schema expects it
    payload = {
        "Machine_Type": machine_type,
        "Air_Temperature_K": air_temp,
        "Process_Temperature_K": process_temp,
        "Rotational_Speed_rpm": rpm,
        "Torque_Nm": torque,
        "Tool_Wear_min": tool_wear
    }

    try:
        # Send the JSON packet over the network to FastAPI
        with st.spinner("Communicating with central AI engine..."):
            response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            prediction = result["critical_failure_imminent"]
            probability = result["failure_probability"]

            # Display the outputs
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability * 100,
                    number = {'suffix': "%"},
                    title = {'text': "Failure Probability Risk"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if prediction else "darkgreen"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "lightyellow"},
                            {'range': [70, 100], 'color': "salmon"}
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Engine Diagnostics Response")
                if prediction:
                    st.error("⚠️ **CRITICAL:** API returned failure notification. Action required.")
                else:
                    st.success("✅ **NORMAL:** API confirms parameters are within safe operation limits.")
        else:
            st.error(f"Backend API returned an error: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Connection Failed! Make sure your FastAPI server is running (`uvicorn api:app --reload`) in your terminal.")

else:
    st.info("👈 Adjust parameters and click **Run Live Diagnostics** to ping the API.")
