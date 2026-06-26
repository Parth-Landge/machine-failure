#  AI-Powered Predictive Maintenance System

An event-driven, full-stack IoT architecture that predicts industrial machine failures and generates real-time, context-aware repair strategies using Generative AI (RAG).

---

##  Project Overview

This project simulates an industrial predictive maintenance ecosystem. Instead of a monolithic codebase, it utilizes a professional decoupled architecture:

- **The Brain (Backend):** A Scikit-Learn Random Forest model exposed via a FastAPI endpoint hosted on Render.
- **The Face (Frontend):** An interactive Streamlit dashboard that allows users to simulate live sensor data.
- **The AI Mechanic (RAG):** When a critical failure is predicted, the system uses Google Gemini (Generative AI) to read a local machine manual and generate step-by-step emergency repair instructions based on the exact live sensor anomalies.

---

##  Repository Structure (File Architecture)

Every file in this repository has a distinct, professional role in the MLOps pipeline:

-  **app.py:** The Streamlit frontend. Acts as the client, capturing simulated sensor inputs and displaying the RAG-generated repair guides.

-  **api.py:** The FastAPI backend. Loads the trained model and serves predictions via HTTP POST requests.

-  **train_with_mlflow.py:** The MLOps training script. Trains the Random Forest model and uses MLflow to track hyperparameters, accuracy metrics, and model artifacts in a local SQLite database.

- **machine_manual.txt:** The knowledge base. A text file acting as the factory maintenance manual, consumed by the Gemini LLM for Retrieval-Augmented Generation (RAG).

-  **model.pkl:** The serialized, production-ready machine learning model.

- **requirements.txt:** Dependency list for seamless cloud deployment.

---

## How to Run Locally

### Clone the repository

```bash
git clone https://github.com/parth_landge/machine-failure.git
cd machine-failure
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### View MLOps Tracking (Optional)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Run the Streamlit Dashboard

```bash
streamlit run app.py
```

---

## Skills Demonstrated

Through building this project, I engineered a complete end-to-end data lifecycle:

- **MLOps:** Implementing MLflow for hyperparameter tracking and model versioning.

- **Cloud Architecture:** Decoupling frontend (Streamlit) from backend APIs (FastAPI/Render).

- **Generative AI Integration:** Building an event-driven RAG system using Google's `google-generativeai` SDK to give the mathematical model a "voice."

---

##  Future Scope & Real-World Application

Currently, the system relies on manual data input via the Streamlit dashboard. The next engineering phase involves Hardware Integration:

- Utilizing an ESP32 Microcontroller equipped with DHT22 (Temperature) and IR (Rotational Speed) sensors.

- Streaming live, physical hardware data via Wi-Fi directly to the FastAPI endpoint to monitor a real-world milling machine or desk motor.
