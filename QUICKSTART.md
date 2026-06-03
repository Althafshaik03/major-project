# Quickstart Guide

This is a simplified guide to quickly getting the project up and running for evaluation or demonstration.

## Prerequisites
- **Python 3.10+** (Ensure it is added to your PATH)
- **Node.js** (Required for the frontend dashboard)

## Step 1: Start the Backend (API Server)
1. Open a terminal in the root folder of this project (`Major Project`).
2. If you haven't already, install the Python dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Start the FastAPI server using `uvicorn`:
   ```powershell
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
4. Wait until the terminal says `Application startup complete`. (First-time startup might take a minute as it loads the simulated dataset).

## Step 2: Start the Frontend (React Dashboard)
1. Open a **second** terminal and navigate to the frontend folder:
   ```powershell
   cd frontend/app
   ```
2. Install the necessary Node packages (only needed the first time):
   ```powershell
   npm install
   ```
3. Start the Vite development server:
   ```powershell
   npm run dev
   ```

## Step 3: View the Dashboard
1. Open your web browser and go to: **http://localhost:5173**
2. You will now see the Live Ventilator Dashboard, which is connected to the Python backend running on port 8000.

---

### Optional: Train Models or Run Tests
If you want to re-train the AI models or run the evaluation pipeline, open a terminal in the project root and run:

- **Generate Synthetic Data:** `python pipelines/run_phase1.py`
- **Train the Dual-Head LSTM:** `python ml/lstm_training.py`
- **Train the Multi-Risk Model:** `python ml/multi_risk_training.py`
- **Run the Digital Twin Gate:** `python pipelines/evaluate_digital_twin.py`
- **Run the Python Unit Tests:** `python -m unittest discover -s tests -p "test_*.py"`

For more detailed deployment instructions (like running the Grafana/Prometheus Docker stack), please check out the `RUNNING.md` file.
