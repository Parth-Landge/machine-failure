import streamlit as st
import plotly.graph_objects as go
import requests
import google.generativeai as genai
import os

# 1. Page Configuration
st.set_page_config(page_title="Predictive Maintenance", layout="wide")

API_URL = "https://milling-machine-api.onrender.com/predict"

# 2. Sidebar Controls
st.sidebar.header("🔑 AI Access")
gemini_api_key = st.sidebar.text_input("Google Gemini API Key", type="password")

st.sidebar.header("🛠️ Machine Settings")
machine_type = st.sidebar.selectbox("Machine Quality Type", ["L", "M", "H"])
air_temp = st.sidebar.number_input("Air Temperature [K]", value=298.0)
process_temp = st.sidebar.number_input("Process Temperature [K]", value=308.0)
rpm = st.sidebar.number_input("Rotational Speed [rpm]", value=1500.0)
torque = st.sidebar.number_input("Torque [Nm]", value=40.0)
tool_wear = st.sidebar.number_input("Tool Wear [min]", value=50.0)

# 3. Main Dashboard Area
st.title("🏭 AI-Powered Predictive Maintenance")
st.write("Live sensor monitoring with Generative AI RAG Diagnostics.")

predict_button = st.sidebar.button("Run Live Diagnostics", use_container_width=True)

if predict_button:
    payload = {
        "Machine_Type": machine_type,
        "Air_Temperature_K": air_temp,
        "Process_Temperature_K": process_temp,
        "Rotational_Speed_rpm": rpm,
        "Torque_Nm": torque,
        "Tool_Wear_min": tool_wear
    }

    try:
        with st.spinner("Communicating with central API engine..."):
            response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            prediction = result["critical_failure_imminent"]
            probability = result["failure_probability"]

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
                if not prediction:
                    st.success("✅ **NORMAL:** API confirms parameters are within safe operation limits.")
                else:
                    st.error("⚠️ **CRITICAL FAILURE IMMINENT:** Machinery limits exceeded.")
                    
                    # --- THE RAG AI MECHANIC STARTS HERE ---
                    st.markdown("---")
                    st.subheader("🤖 AI Mechanic Repair Guide")
                    
                    if not gemini_api_key:
                        st.warning("Please enter your Gemini API Key in the sidebar to generate a custom repair guide.")
                    else:
                        with st.spinner("AI Mechanic is reading the manual and writing a repair strategy..."):
                            try:
                                # 1. Configure the AI
                                genai.configure(api_key=gemini_api_key)
                                model = genai.GenerativeModel('gemini-2.5-flash')
                                
                                # 2. Read the Local Manual
                                with open("machine_manual.txt", "r") as file:
                                    manual_context = file.read()
                                    
                                # 3. Write the Prompt for the LLM
                                prompt = f"""
                                You are an expert factory mechanic. The predictive maintenance AI just triggered a failure alert.
                                Here is the live sensor data that caused the failure:
                                - Process Temp: {process_temp} K
                                - RPM: {rpm}
                                - Torque: {torque} Nm
                                - Tool Wear: {tool_wear} mins
                                
                                Based strictly on the official maintenance manual below, write a short, 3-step emergency repair guide for the operator.
                                
                                MANUAL:
                                {manual_context}
                                """
                                
                                # 4. Generate and display the answer!
                                ai_response = model.generate_content(prompt)
                                st.info(ai_response.text)
                                
                            except Exception as e:
                                st.error(f"AI Generation Failed: {e}")

        else:
            st.error(f"Backend API returned an error: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Connection Failed! Make sure your API is running.")
else:
    st.info("👈 Adjust parameters and click **Run Live Diagnostics**.")
