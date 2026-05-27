# 🫁 Blockchain-Enabled Digital Twin Framework for Ventilator Parameter Optimization

> A real-time, AI-driven clinical co-pilot for ICU mechanical ventilation — combining a **Digital Twin**, **Dual-Head LSTM Forecaster**, **Multi-Risk LSTM**, **PPO Reinforcement Learning Agent**, and a **Blockchain Audit Trail** — built as a final-year B.Tech major project.

![Status](https://img.shields.io/badge/status-Phase%208%20%E2%80%93%20Final%20Packaging-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![TensorFlow](https://img.shields.io/badge/ML-TensorFlow%2FKeras-FF6F00)
![Solidity](https://img.shields.io/badge/Blockchain-Solidity%2FHardhat-363636)
![License](https://img.shields.io/badge/license-Academic-lightgrey)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [What This Project Does](#-what-this-project-does)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Module Breakdown](#-module-breakdown)
  - [Data Simulator](#1-data-simulator-servicesdatasimulatorpy)
  - [Digital Twin](#2-digital-twin-servicesdigital_twinpy)
  - [LSTM Forecasting Engine](#3-lstm-forecasting-engine)
  - [Multi-Risk LSTM](#4-multi-risk-lstm-mlmulti_risk_trainingpy)
  - [PPO RL Agent](#5-ppo-reinforcement-learning-agent)
  - [Blockchain Audit Layer](#6-blockchain-audit-layer)
  - [FastAPI Backend](#7-fastapi-backend-apimainpy)
  - [React Frontend Dashboard](#8-react-frontend-dashboard)
  - [Observability Stack](#9-observability-stack)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start Guide](#-quick-start-guide)
- [API Reference](#-api-reference)
- [Running Tests](#-running-tests)
- [ML Training Pipeline](#-ml-training-pipeline)
- [Blockchain Setup](#-blockchain-setup)
- [Performance KPIs](#-performance-kpis)
- [Documentation Index](#-documentation-index)
- [Implementation Phases](#-implementation-phases)
- [Known Limitations](#-known-limitations)
- [Authors](#-authors)

---

## 🏥 Problem Statement

Mechanical ventilators in ICUs are manually tuned by clinical staff with **limited foresight** into how parameter changes affect patient respiratory state. This creates risks including:

- **Ventilator-Induced Lung Injury (VILI)** from incorrect tidal volume settings
- **Hypoxia events** from suboptimal FiO₂ / PEEP combinations
- **Delayed responses** due to lack of predictive tools
- **No audit trail** for clinical decisions made by ventilator systems

This project addresses all four challenges with an end-to-end AI + blockchain platform.

---

## 🎯 What This Project Does

This system is an **end-to-end clinical co-pilot** that:

1. **Ingests** live ventilator telemetry (real or simulated — HR, MAP, SpO₂, PEEP, FiO₂, TidalVol, RespRate)
2. **Forecasts** SpO₂ trajectories and detects hypoxia/desaturation risk using an **LSTM** model (dual-head + multi-risk variant)
3. **Predicts 5 clinical risks** simultaneously (Hypoxia, Tachycardia, Hypotension, Tachypnea, VILI) with next-step vital forecasts
4. **Recommends** PEEP / FiO₂ / TidalVol adjustments using a **PPO RL agent** trained inside a **Digital Twin** environment with clinical safety guards
5. **Simulates** what-if scenarios through the Digital Twin before any recommendation is surfaced to clinicians
6. **Anchors** every recommendation event on an **immutable blockchain audit ledger** (off-chain SHA-256 hash chain + on-chain Solidity anchor)
7. **Visualizes** live patient state, predictions, risk alerts, and audit trail on a real-time React dashboard with **Prometheus + Grafana** observability

---

## 🏗 System Architecture

```
                      ┌─────────────────────────────────────┐
                      │  Frontend Dashboard (React/Vite)      │
                      │  Live patient vitals + LSTM forecasts │
                      │  PPO recommendations + risk gauges    │
                      └──────────────┬──────────────────────┘
                                     │ REST/JSON (port 5173 → 8000)
                                     ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │                    FastAPI Service  (api/main.py : 8000)              │
   │  /recommend  /risks  /twin/replay  /simulator  /audit  /metrics       │
   └─────┬────────────┬───────────────┬─────────────┬────────────┬──────┘
         │            │               │             │            │
         ▼            ▼               ▼             ▼            ▼
   Data        Digital Twin     LSTM Inference  Multi-Risk   PPO Policy
   Simulator   (services/       (services/      Inference    (services/
   (services/  digital_twin)    lstm_inference) (services/   ppo_policy)
   data_sim)                                    multi_risk)
                    │               │              │            │
                    └───────────────┴──────────────┴────────────┘
                                          │
                                    Audit Bridge
                                  (services/audit_bridge)
                                          │
                         ┌────────────────┴──────────────────┐
                         ▼                                   ▼
              SQLite Hash-Chain (off-chain)     AuditAnchor.sol (on-chain)
              blockchain/audit_ledger.db        Hardhat / web3.py bridge

   ┌─────────────────────────────────────┐
   │  Observability (deploy/)             │
   │  Prometheus (port 9090)              │
   │  Grafana (port 3000)                 │
   │  /metrics scrape endpoint            │
   └─────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔁 **Digital Twin** | Physiologically bounded patient model — calibrates from patient history, simulates what-if scenarios, enforces safety clamps |
| 🧠 **Dual-Head LSTM** | Predicts next-step SpO₂ (regression) + hypoxia risk (classification) from 12-step ventilator sequences |
| 🎯 **Multi-Risk LSTM** | Multi-task BiLSTM with 10 outputs — 5 vital forecasts + 5 clinical risk probabilities in a single forward pass |
| 🤖 **PPO RL Agent** | Stable-Baselines3 trained in a custom `VentilatorTwinEnv` gymnasium environment; a rule-based safety layer prevents any unsafe recommendations |
| 🔐 **Blockchain Audit** | SHA-256 off-chain hash chain (SQLite) + on-chain Solidity `AuditAnchor` contract deployed via Hardhat |
| 🌦 **Weather Modulation** | Atmospheric conditions (pressure, humidity, temperature) modulate effective FiO₂ efficiency and SpO₂ baseline in the twin |
| 📈 **Observability** | `/metrics` for Prometheus scraping; pre-provisioned Grafana dashboard visualizes observed vs. predicted SpO₂ |
| 🧪 **12-Test Suite** | Unit + integration tests covering simulator API, digital twin safety, blockchain anchoring, multi-risk inference, weather modulation |
| 📦 **One-command Pipeline** | `python pipelines/run_phase1.py` generates synthetic data + feature artifacts end-to-end |

---

## 🔧 Module Breakdown

### 1. Data Simulator (`services/data_simulator.py`)

Generates realistic synthetic ventilator telemetry for patients across four disease profiles:

| Profile | Description |
|---------|-------------|
| `normal` | Stable respiratory mechanics, SpO₂ 95–100% |
| `ards` | Severe ARDS — low compliance, high FiO₂ requirements |
| `copd` | Chronic obstructive — elevated RespRate, baseline hypercapnia risk |
| `unstable` | Hemodynamically unstable — drifting SpO₂, high hypoxia rate |
| `lung_infected` | Lung infection profile — reduced SpO₂ / elevated HR and RR |

**Configurable behaviors via `SimulationConfig`:**
- `interval_minutes` — inter-reading time interval
- `packet_loss_probability` — simulates dropped telemetry packets
- `artifact_probability` — random spike/dropout artifact injection
- `trend_strength` — strength of underlying clinical deterioration drift
- `seed` — reproducibility seed

**Outputs per record:** `HR`, `MAP`, `RespRate`, `SpO₂`, `PEEP`, `FiO₂`, `TidalVol`, `timestamp`, `stay_id`

---

### 2. Digital Twin (`services/digital_twin.py`)

A **physiologically bounded respiratory mechanics model** that maps ventilator settings → predicted SpO₂ trajectory.

**Physics model:**
- `PEEP` provides alveolar recruitment → raises mean SpO₂ (coefficient: 0.35 per cmH₂O delta)
- `FiO₂` is the primary oxygen driver (coefficient: 0.18 per % delta)
- `TidalVol` above 450 mL optimal incurs a VILI penalty
- **Mean reversion (45%)** prevents overshoot — the twin blends the step with the calibrated baseline
- Includes **breath-cycle sinusoidal modulation** and **perfusion factor** from MAP
- Optional **weather state** modulates FiO₂ efficiency via atmospheric pressure

**Calibration flow:**
```python
twin = DigitalTwin(stay_id=30004018)
twin.calibrate(history_records)   # fits compliance_factor, baseline_spo2, uncertainty
result = twin.simulate(
    proposed={'PEEP': 10, 'FiO2': 70, 'TidalVol': 400},
    current_spo2=94.0,
    steps=4,          # 4 × 15-min steps = 1 hour
    noise_scale=1.0
)
```

**Safety bounds (hard-clamp):**
| Parameter | Min | Max |
|-----------|-----|-----|
| PEEP | 3.0 cmH₂O | 20.0 cmH₂O |
| FiO₂ | 21% | 100% |
| TidalVol | 200 mL | 800 mL |

---

### 3. LSTM Forecasting Engine

#### Single-Task (Dual-Head) LSTM — `ml/lstm_training.py` + `services/lstm_inference.py`

- **Input**: 12-step sequence × 35 engineered features (lags, rolling stats, ICU ratios)
- **Outputs**:
  - Head 1 (regression): `Next_SpO2` — predicted SpO₂ 15 min ahead
  - Head 2 (classification): `Hypoxia_Risk` — probability of SpO₂ < 90% in next step
- **Training source**: Synthetic + MIMIC-like historical data via Phase 1 pipeline
- **Artifacts**: `.keras` model file + `.pkl` feature scalers stored under `ml/`

#### Fallback behavior
When no trained Keras model is found, the API falls back to **heuristic predictions** (rule-based SpO₂ estimate). The `/health` endpoint reports which mode is active.

---

### 4. Multi-Risk LSTM (`ml/multi_risk_training.py`)

A more advanced **multi-task BiLSTM** model with 10 simultaneous outputs:

**Architecture:**
```
Input  [12 timesteps × 102 features]
  ↓
BiLSTM Layer 1: 256 units (return_sequences=True) + LayerNorm
  ↓
BiLSTM Layer 2: 128 units + BatchNorm + Dense 128 (ReLU) + Dropout 0.4
  ↓
Shared Dense: 64 units (ReLU) + Dropout 0.4
  ↓ (splits into 10 task-specific heads)
  ├── 5 Regression Heads  → Next_SpO2, Next_HR, Next_MAP, Next_RespRate, Next_TidalVol
  └── 5 Classification Heads → Hypoxia_Risk, Tachycardia_Risk, Hypotension_Risk, Tachypnea_Risk, VILI_Risk
```

**Model stats:**
- Parameters: 1,497,546 (~5.7 MB)
- Training data: 37,984 sequences (from 799,964 records across 4,566 patient stays)
- Optimizer: Adam (lr=0.001), Focal Loss (γ=1.5) for imbalanced classifications

**Clinical risk thresholds:**
| Risk | Clinical Condition | Threshold |
|------|-------------------|-----------|
| Hypoxia_Risk | SpO₂ < 90% | ICU safety guideline |
| Tachycardia_Risk | HR < 40 or > 140 bpm | Cardiac stability |
| Hypotension_Risk | MAP < 60 mmHg | Organ perfusion minimum |
| Tachypnea_Risk | RR < 8 or > 30 breaths/min | Respiratory effort limits |
| VILI_Risk | TidalVol < 280 or > 600 mL | Ventilator-induced injury prevention |

**Risk stratification guide:**
| Level | Criteria | Action |
|-------|----------|--------|
| 🟢 Green | All risks < 0.3 | Routine monitoring |
| 🟡 Yellow | 1–2 risks in [0.3–0.7] | Increase observation every 15–30 min |
| 🟠 Orange | 2+ risks in [0.5–0.8] OR 1 risk > 0.7 | Active intervention, call RT |
| 🔴 Red | 3+ risks > 0.7 OR any risk > 0.9 | 🚨 Immediate physician contact |

---

### 5. PPO Reinforcement Learning Agent

**Training environment** — `ml/ppo_training.py` defines `VentilatorTwinEnv(gymnasium.Env)`:
- **State space**: Current vitals + digital twin trajectory + LSTM forecast
- **Action space**: `Discrete(9)` — clinician-style PEEP/FiO₂/TidalVol adjustments (±small steps)
- **Reward shaping**:
  - Penalizes SpO₂ < 90% (hypoxia)
  - Penalizes TidalVol > 600 mL (VILI risk)
  - Penalizes high PEEP (barotrauma)
  - Rewards movement toward safe SpO₂ range (94–99%)

**Rule-based safety layer** (`services/ppo_policy.py`):  
Before returning any PPO recommendation, a deterministic safety validator re-clamps all values and overrides with conservative defaults if the RL policy proposes out-of-bound settings.

**Outputs:** `recommended_PEEP`, `recommended_FiO2`, `recommended_TidalVol` + confidence score

---

### 6. Blockchain Audit Layer

#### Off-Chain Hash Chain — `services/audit_bridge.py`
- Every clinical event (recommendation, risk prediction, twin replay) is serialized + SHA-256 hashed
- Each block links to the previous block hash (tamper-evident chain)
- Stored in a local **SQLite database** (`blockchain/audit_ledger.db`)
- Events include: `stay_id`, `event_type`, `actor`, `payload`, `timestamp`, `block_hash`, `prev_hash`

#### On-Chain Anchor — `blockchain/contracts/AuditAnchor.sol` + `services/chain_anchor.py`
- **Solidity smart contract**: `AuditAnchor.sol`
  - Append-only block registry on Ethereum-compatible chain
  - Owner-rotatable writer allowlist
  - Contiguity-enforced anchoring (each anchor must reference previous on-chain hash)
- **Hardhat** development environment with JavaScript test suite (`blockchain/test/AuditAnchor.test.js`)
- **Python bridge** (`services/chain_anchor.py`) batches pending off-chain blocks and commits them on-chain via `web3.py`
- API endpoint `POST /audit/anchor` exposes `dry_run` / `live` modes

---

### 7. FastAPI Backend (`api/main.py`)

The central orchestration layer. All services are initialized at startup and exposed via REST endpoints.

**Key Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root — links to all endpoints |
| `GET` | `/health` | Service health + LSTM artifact status |
| `GET` | `/docs` | Swagger UI (auto-generated) |
| `POST` | `/simulator/session/{stay_id}` | Create a simulator session |
| `GET` | `/simulator/session/{key}/next` | Fetch next simulated telemetry record |
| `GET` | `/simulator/session/{key}/batch` | Fetch a batch of records (up to 512) |
| `POST` | `/twin/replay` | Run deterministic or seeded twin simulation |
| `POST` | `/patient/{stay_id}/recommend` | Full pipeline: twin + LSTM + PPO → recommendation |
| `POST` | `/patient/{stay_id}/risks` | Multi-risk LSTM inference (5 vitals + 5 risks) |
| `POST` | `/audit/anchor` | Blockchain batch anchor (dry_run or live) |
| `GET` | `/metrics` | Prometheus metrics endpoint (for Grafana) |

**Startup sequence:**
1. Load historical dataset (`clean_full_data_v2.csv` → `data/simulated_phase1.csv` → generate demo data)
2. Load LSTM inference engine (Keras model + scalers)
3. Load Multi-Risk inference engine
4. Initialize Audit Bridge
5. Register Prometheus metrics

---

### 8. React Frontend Dashboard

Located in `frontend/app/` — a **React + Vite** single-page application:

- **Live SpO₂ chart**: Observed vs. LSTM-predicted next SpO₂ (updates every ~5 s)
- **Hypoxia probability gauge**: Real-time risk percentage
- **Multi-risk panel**: 5 risk gauges (Hypoxia, Tachycardia, Hypotension, Tachypnea, VILI)
- **PPO Recommendation card**: Current recommended PEEP / FiO₂ / TidalVol
- **Digital Twin simulation preview**: Simulated SpO₂ trajectory for proposed settings
- **Blockchain audit trail**: Latest block hash + event history

**Dependencies:** React, Vite, Recharts (charting), Three.js (optional 3D visualization)

---

### 9. Observability Stack

Located in `deploy/` — Docker Compose managed:

```
deploy/
├── docker-compose.yml          # Prometheus + Grafana containers
├── prometheus/
│   └── prometheus.yml          # Scrape config → host.docker.internal:8000/metrics
└── grafana/
    ├── provisioning/
    │   ├── datasources/        # Auto-configured Prometheus data source
    │   └── dashboards/         # Auto-provisioned dashboard JSON
    └── dashboards/
        └── ventilator_lstm.json  # SpO₂ vs LSTM forecast, hypoxia risk panels
```

**Prometheus gauges (updated per `/recommend` call):**
- `ventilator_spo2_current` — observed SpO₂
- `ventilator_spo2_predicted` — LSTM-predicted next SpO₂
- `ventilator_hypoxia_prob` — hypoxia probability
- `ventilator_lstm_keras_used` — 1 if Keras model was used, 0 if heuristic fallback

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| **API** | FastAPI, Uvicorn, Pydantic, Starlette |
| **ML / Forecasting** | TensorFlow / Keras (LSTM + BiLSTM), scikit-learn |
| **Reinforcement Learning** | Stable-Baselines3 (PPO), Gymnasium custom env |
| **Blockchain** | Solidity (AuditAnchor.sol), Hardhat, web3.py |
| **Storage** | SQLite (audit ledger), CSV / pickled feature artifacts |
| **Observability** | Prometheus, Grafana (Docker Compose) |
| **Frontend** | React, Vite, Recharts, Three.js |
| **Testing** | unittest, pytest, FastAPI TestClient, httpx |
| **Language** | Python 3.10+, JavaScript (Node.js) |

---

## 📁 Project Structure

```
Major Project/
│
├── api/
│   └── main.py                  # FastAPI app — all REST endpoints (28K bytes)
│
├── services/                    # Core domain services
│   ├── digital_twin.py          # Patient-specific twin model
│   ├── lstm_inference.py        # Dual-head LSTM inference engine
│   ├── multi_risk_inference.py  # Multi-task LSTM inference engine
│   ├── ppo_policy.py            # PPO recommendation layer + rule-based safety
│   ├── audit_bridge.py          # Off-chain SHA-256 hash chain (SQLite)
│   ├── chain_anchor.py          # On-chain Solidity bridge (web3.py)
│   ├── data_simulator.py        # Ventilator telemetry simulator
│   ├── prometheus_metrics.py    # Prometheus gauge definitions
│   ├── weather.py               # Weather state → FiO₂ efficiency / SpO₂ penalty
│   └── fiware_adapter.py        # FIWARE/NGSI-LD adapter (optional IoT integration)
│
├── ml/                          # Training scripts + artifacts (gitignored)
│   ├── lstm_training.py         # Dual-head LSTM trainer
│   ├── multi_risk_training.py   # Multi-task BiLSTM trainer
│   ├── ppo_training.py          # PPO env + SB3 trainer
│   ├── simulated_phase1/        # Feature artifacts from Phase 1
│   └── multi_risk/              # Multi-risk model + artifacts
│       └── multi_risk_lstm.keras  (5.7 MB trained model)
│
├── pipelines/                   # Feature engineering + evaluation pipelines
│   ├── run_phase1.py            # One-command: simulate + engineer features
│   ├── feature_engineering.py   # Windowed feature extraction (35-feature)
│   ├── multi_risk_features.py   # 102-feature multi-risk pipeline
│   ├── simulated_ingestion.py   # Multi-profile dataset generation
│   ├── evaluate_digital_twin.py # CI-style twin quality gate
│   ├── historical_replay_benchmark.py  # Replay vs history benchmark
│   ├── lstm_dataset_size_study.py      # Dataset size ablation
│   ├── lstm_feature_enrichment.py      # Feature enrichment study
│   └── ppo_feature_engineering.py      # PPO state features
│
├── blockchain/
│   ├── contracts/
│   │   └── AuditAnchor.sol      # Solidity audit smart contract
│   ├── scripts/                 # Hardhat deploy scripts
│   ├── test/
│   │   └── AuditAnchor.test.js  # Solidity contract tests
│   ├── hardhat.config.js        # Hardhat configuration
│   ├── package.json
│   └── audit_ledger.db          # Off-chain SQLite hash chain
│
├── frontend/
│   └── app/                     # React + Vite dashboard
│       ├── src/                 # React source files
│       ├── index.html
│       └── package.json
│
├── deploy/                      # Docker Compose observability stack
│   ├── docker-compose.yml
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       └── provisioning/
│
├── tests/                       # Unit + integration test suite
│   ├── test_simulator_api.py
│   ├── test_digital_twin_replay.py
│   ├── test_digital_twin_safety.py
│   ├── test_multi_risk.py
│   ├── test_chain_anchor.py
│   ├── test_ppo_training_smoke.py
│   ├── test_weather_modulation.py
│   ├── test_lung_infected_profile.py
│   ├── test_lstm_size_study.py
│   ├── test_scenarios.py
│   └── test_scenario_outputs.py
│
├── docs/                        # Documentation, diagrams, papers
│   ├── architecture-decisions.md
│   ├── twin-model-spec.md
│   ├── event-schema.md
│   ├── safety-constraints.md
│   ├── multi_risk_integration_guide.md
│   ├── blockchain_ventilator_framework.md
│   ├── failure-recovery.md
│   ├── demo-runbook.md
│   ├── diagrams/               # Mermaid DFD, UML, architecture diagrams
│   └── presentation/           # Slides, viva prep, demo scripts
│
├── reports/                     # Model evaluation + ablation reports
│   ├── model-evaluation-twin.md
│   ├── model-evaluation-lstm.md
│   ├── model-evaluation-ppo.md
│   ├── model_evaluation_multi_risk.md
│   ├── ablation-study.md
│   └── benchmark-results.md
│
├── data/                        # Generated datasets (gitignored by default)
│   └── simulated_phase1.csv
│
├── requirements.txt             # Python dependencies
├── README.md                    # ← You are here
├── RUNNING.md                   # Detailed local run guide
├── IMPLEMENTATION_LOG.md        # Phase-by-phase implementation journal
├── README_MULTI_RISK.md         # Multi-risk module quick reference
└── MULTI_RISK_QUICK_REFERENCE.md
```

---

## 🚀 Quick Start Guide

> **Full guide with troubleshooting:** [RUNNING.md](RUNNING.md)

### Prerequisites

- **Python 3.10+** (3.11 or 3.12 recommended)
- **Node.js 18+** (for frontend and Hardhat)
- **Docker Desktop** *(optional — only for Grafana + Prometheus)*

### Step 1 — Install Python Dependencies

```powershell
# From the project root: "Major Project/"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 2 — Run the API Server

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Sanity check URLs:
- Root: http://127.0.0.1:8000/
- Health: http://127.0.0.1:8000/health
- Swagger UI: http://127.0.0.1:8000/docs

> **Note:** On first launch without a dataset, the API auto-generates `data/demo_ventilator_data.csv` so the dashboard works immediately.

### Step 3 — Generate Synthetic Data + Train LSTM *(recommended)*

```powershell
# Phase 1: generate data + engineer features
python pipelines/run_phase1.py

# Train the dual-head LSTM
$env:LSTM_ARTIFACTS_DIR = "$(Resolve-Path '.\ml\simulated_phase1')"
python ml/lstm_training.py
```

Restart the API with the same env variable to activate the Keras model.

### Step 4 — Train Multi-Risk LSTM *(optional, requires historical MIMIC-like CSV)*

```powershell
# Feature engineering for multi-risk (102 features, 5 targets)
python -m pipelines.multi_risk_features --max-patients 400

# Train multi-task BiLSTM
python ml/multi_risk_training.py
```

### Step 5 — Open the React Dashboard

```powershell
cd frontend/app
npm install        # first time only
npm run dev
# Open: http://127.0.0.1:5173
```

### Step 6 — Start Grafana + Prometheus *(optional)*

```powershell
cd deploy
docker compose up -d
# Grafana:    http://localhost:3000  (admin / admin)
# Prometheus: http://localhost:9090
```

---

## 📡 API Reference

### Recommendation Pipeline

```bash
POST /patient/{stay_id}/recommend
Content-Type: application/json

{
  "history": [
    {"HR": 88, "MAP": 75, "RespRate": 18, "SpO2": 94, "PEEP": 8, "FiO2": 55, "TidalVol": 450},
    ...   # minimum 1 record; 12+ records enables Keras LSTM + twin calibration
  ]
}
```

**Response:**
```json
{
  "stay_id": 30004018,
  "recommended": {"PEEP": 9.0, "FiO2": 58.0, "TidalVol": 440.0},
  "twin_simulation": {
    "trajectory": [94.0, 95.2, 95.8, 96.1, 96.3],
    "mean_spo2": 95.85,
    "delta_spo2": 1.85,
    "risk_flag": false
  },
  "lstm_forecast": {
    "pred_next_spo2": 95.4,
    "hypoxia_prob": 0.03,
    "lstm_forecast_source": "lstm_keras"
  },
  "audit_hash": "a3f7c2e1..."
}
```

### Multi-Risk Inference

```bash
POST /patient/{stay_id}/risks
Content-Type: application/json

{
  "history": [
    {"HR": 85, "MAP": 75, "RespRate": 18, "SpO2": 96, "PEEP": 5, "FiO2": 40, "TidalVol": 450},
    ...   # 12 records required
  ]
}
```

**Response:**
```json
{
  "stay_id": 30004018,
  "predictions": {
    "regression": {
      "Next_SpO2":     {"prediction": 96.5},
      "Next_HR":       {"prediction": 87.2},
      "Next_MAP":      {"prediction": 76.1},
      "Next_RespRate": {"prediction": 18.4},
      "Next_TidalVol": {"prediction": 453.0}
    },
    "classification": {
      "Hypoxia_Risk":      {"probability": 0.04, "risk": 0},
      "Tachycardia_Risk":  {"probability": 0.02, "risk": 0},
      "Hypotension_Risk":  {"probability": 0.07, "risk": 0},
      "Tachypnea_Risk":    {"probability": 0.05, "risk": 0},
      "VILI_Risk":         {"probability": 0.12, "risk": 0}
    }
  },
  "summary": {"high_risk_flags": [], "next_spo2": 96.5},
  "source": "multi_risk_lstm"
}
```

### Digital Twin Replay

```bash
POST /twin/replay
Content-Type: application/json

{
  "stay_id": 910050,
  "proposed": {"PEEP": 10, "FiO2": 65, "TidalVol": 430},
  "steps": 4,
  "noise_scale": 0    # 0 = deterministic replay
}
```

### Blockchain Anchor

```bash
POST /audit/anchor
Content-Type: application/json

{"mode": "dry_run"}   # or "live" for on-chain submission
```

---

## 🧪 Running Tests

```powershell
# Full test suite
python -m unittest discover -s tests -p "test_*.py"

# Or with pytest
python -m pytest tests/ -v

# Digital twin CI quality gate
python pipelines/evaluate_digital_twin.py --fail-on-thresholds
```

### Test Coverage

| Test File | What It Covers |
|-----------|----------------|
| `test_simulator_api.py` | Session creation, next/batch record endpoints, schema validation |
| `test_digital_twin_replay.py` | Deterministic replay, seeded stochastic, safe-bound clamping |
| `test_digital_twin_safety.py` | Extreme parameter inputs, safety clamp assertions, VILI flag |
| `test_multi_risk.py` | Multi-risk inference, API endpoint, response structure |
| `test_chain_anchor.py` | Hash chain construction, block linkage, off-chain integrity |
| `test_ppo_training_smoke.py` | PPO trainer smoke test (runs without SB3/gymnasium) |
| `test_weather_modulation.py` | Calm vs storm vs high-altitude → divergent SpO₂ trajectories |
| `test_lung_infected_profile.py` | Lung-infected simulator profile + clinical assertions |
| `test_lstm_size_study.py` | LSTM MAE/RMSE/F1 across dataset size variants |
| `test_scenarios.py` | End-to-end clinical scenario outputs |

---

## 🤖 ML Training Pipeline

### Phase 1 — Synthetic Data + Dual-Head LSTM

```powershell
# Step 1: One-command data generation + feature engineering
python pipelines/run_phase1.py
# Options: --stays-per-profile 10 --steps-per-stay 96 --seed 42 --seq-len 12

# Step 2: Train dual-head LSTM
$env:LSTM_ARTIFACTS_DIR = "$(Resolve-Path '.\ml\simulated_phase1')"
python ml/lstm_training.py
```

### Multi-Risk LSTM Pipeline

```powershell
# Step 1: Feature engineering (102 features, multi-profile dataset)
python -m pipelines.multi_risk_features --max-patients 400

# Step 2: Train multi-task BiLSTM (15–40 epochs with early stopping)
python ml/multi_risk_training.py

# Optional: Hyperparameter overrides
$env:LSTM_EPOCHS = "20"
$env:LSTM_BATCH_SIZE = "256"
$env:LSTM_LR = "0.001"
python ml/multi_risk_training.py
```

### PPO Agent Training

```powershell
# Smoke test (no SB3 required)
python ml/ppo_training.py --smoke

# Full training (requires stable-baselines3 + gymnasium)
python ml/ppo_training.py
```

### Dataset Size Ablation Study

```powershell
# Laptop demo (a few minutes per size)
python pipelines/lstm_dataset_size_study.py --sizes 1000,2000 --epochs 8

# Paper-grade run
python pipelines/lstm_dataset_size_study.py --sizes 1000,2000,4000,8000 --epochs 20
```

---

## ⛓ Blockchain Setup

### Off-Chain (Python — works out of the box)

The SQLite audit ledger is created automatically on the first `/audit/anchor` call.

### On-Chain (Hardhat — optional)

```powershell
cd blockchain
npm install

# Run Solidity contract tests
npx hardhat test

# Deploy to local Hardhat network
npx hardhat node           # start local chain (terminal 1)
npx hardhat run scripts/deploy.js --network localhost  # terminal 2
```

Configure `services/chain_anchor.py` with the deployed contract address for live on-chain anchoring.

---

## 📊 Performance KPIs

| KPI | Target | Status |
|-----|--------|--------|
| Inference + recommendation latency | < 2 seconds | ✅ |
| Asynchrony risk model AUROC | > 0.85 | ✅ (Hypoxia AUROC 0.85–0.92) |
| Prediction error improvement over baseline | ≥ 20% | ✅ |
| Audit coverage | 100% of recommendation events | ✅ |
| Digital twin replay consistency | 100% | ✅ |
| Twin trend direction accuracy | ≥ 70% | ✅ (100% after tuning) |
| Twin mean absolute delta SpO₂ | ≤ 8.0 | ✅ (1.495 after tuning) |

---

## 📚 Documentation Index

| Document | Description |
|----------|-------------|
| [RUNNING.md](RUNNING.md) | Step-by-step local run guide (API, dashboard, tests, Grafana) |
| [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md) | Phase-by-phase implementation journal (Phase 0 → 8) |
| [README_MULTI_RISK.md](README_MULTI_RISK.md) | Multi-risk LSTM quick reference |
| [docs/architecture-decisions.md](docs/architecture-decisions.md) | Architecture decision records (ADRs) |
| [docs/twin-model-spec.md](docs/twin-model-spec.md) | Digital twin specification and equations |
| [docs/event-schema.md](docs/event-schema.md) | Canonical telemetry event schema |
| [docs/safety-constraints.md](docs/safety-constraints.md) | Safety guards and clinical bounds |
| [docs/blockchain_ventilator_framework.md](docs/blockchain_ventilator_framework.md) | Blockchain audit design |
| [docs/multi_risk_integration_guide.md](docs/multi_risk_integration_guide.md) | Multi-risk LSTM integration guide |
| [docs/MULTI_RISK_IMPLEMENTATION_SUMMARY.md](docs/MULTI_RISK_IMPLEMENTATION_SUMMARY.md) | Multi-risk implementation summary |
| [docs/failure-recovery.md](docs/failure-recovery.md) | Graceful degradation and recovery |
| [docs/demo-runbook.md](docs/demo-runbook.md) | Step-by-step demo script (3 clinical scenarios) |
| [docs/diagrams/](docs/diagrams/) | DFD, UML, system architecture (Mermaid) |
| [docs/presentation/](docs/presentation/) | Slide notes, viva prep, hindi viva guide |
| [reports/](reports/) | Model evaluation reports (LSTM, PPO, twin, multi-risk, ablation) |

---

## 🗓 Implementation Phases

| Phase | Title | Status | Key Deliverables |
|-------|-------|--------|-----------------|
| **Phase 0** | Project Setup & Governance | ✅ Complete | ADRs, requirements.md, safety-constraints.md |
| **Phase 1** | Data Foundation & Simulation | ✅ Complete | `data_simulator.py`, `run_phase1.py`, event schema |
| **Phase 2** | Digital Twin V1 | ✅ Complete | `digital_twin.py`, twin spec, replay API, CI gate |
| **Phase 3** | LSTM Forecasting Engine | ✅ Complete | `lstm_training.py`, `multi_risk_training.py`, evaluation reports |
| **Phase 4** | PPO Optimization Agent | ✅ Complete | `ppo_training.py`, `VentilatorTwinEnv`, evaluation report |
| **Phase 5** | Blockchain Trust & Audit | ✅ Complete | `AuditAnchor.sol`, `chain_anchor.py`, Hardhat tests |
| **Phase 6** | Integration & Dashboard | ✅ Complete | Full API pipeline, React dashboard, Prometheus/Grafana |
| **Phase 7** | Validation & Benchmarking | ✅ Complete | Benchmark results, ablation study, failure recovery |
| **Phase 8** | Final Packaging | 🔄 In Progress | Report PDF, presentation deck, demo runbook, viva Q&A |

---

## ⚠️ Known Limitations

1. **Academic scope**: Not approved for live production ICU deployment without institutional and regulatory clearance.
2. **15-minute prediction horizon**: Multi-risk model predicts only the next timestep (~15 min ahead).
3. **MIMIC-like training data**: Multi-risk model trained on MIMIC-like synthetic cohort; may require domain adaptation for other hospitals.
4. **No causal inference**: All models are predictive/correlative, not causal.
5. **Imbalanced risk classes**: Rare risks (Hypoxia: 1.73%, Tachycardia: 0.70%) may have lower sensitivity despite Focal Loss.
6. **Artifacts excluded from repo**: Large `.csv`, `.pkl`, `.keras` files are gitignored. See regeneration commands above.
7. **On-chain anchoring**: Requires a local Hardhat node or RPC endpoint; off-chain SQLite works standalone.

---

## 🔄 Regenerating Excluded Artifacts

To keep the repo lean, the following are **gitignored** and must be regenerated locally:

| Artifact | How to Regenerate |
|----------|-------------------|
| `data/*.csv` datasets (~656 MB) | `python pipelines/run_phase1.py` |
| `ml/**/*.pkl` feature tensors (~3.7 GB) | Same as above + `multi_risk_features.py` |
| `*.keras` / `*.h5` trained models | `python ml/lstm_training.py`, `multi_risk_training.py`, `ppo_training.py` |
| Python venv (`.venv/`) | `pip install -r requirements.txt` |
| `blockchain/audit_ledger.db` | Created automatically on first `/audit/anchor` call |

`.gitattributes` is pre-configured for Git LFS (`*.csv`, `*.h5`, `*.keras`, `*.pkl`) if you choose to push artifacts later.

---

## 👤 Authors

**Rishav Kumar** — Final-year B.Tech, 8th Semester Major Project

For viva / academic context see [docs/viva_prep.md](docs/viva_prep.md) and the project report in [docs/final_report/](docs/final_report/).

Reference papers in [docs/](docs/):
- *Improving Patient-Ventilator Synchrony During Pressure Support Ventilation Based on Reinforcement Learning* (IEEE)
- *Machine Learning-Based Digital Twin for Predictive Modeling in Wind Turbines*

---

<div align="center">

**Built with ❤️ for safer ICU ventilation | B.Tech Major Project 2025–2026**

</div>
