# FIWARE Digital Twin (Ventilator Major Project)

Handloom-style architecture adapted for **mechanical ventilation** — same NGSI-v2 pipeline, different domain.

## Architecture

| Layer | Handloom (junior project) | This project (ventilator) |
|-------|---------------------------|---------------------------|
| Sensors | Load cells, MPU6050, IR, DHT22 | ICU CSV / simulator vitals (SpO₂, PEEP, FiO₂, TV, HR, MAP) |
| Context broker | FIWARE Orion + MongoDB | Same (`docker compose` in `ventilator-twin/`) |
| Middleware | Flask relay :5050 | `Fiware/backend/relay.py` |
| 3D | Three.js loom | Three.js **medical-ventilator** (`Ventilator.dae`) |
| AI | Quality / defect / fault | **LSTM** + **PPO** + **physics twin** + **blockchain audit** |

## Dashboards (`Fiware/frontend/`)

| Page | Purpose |
|------|---------|
| [hub.html](Fiware/frontend/hub.html) | Entry + architecture + run instructions |
| [index.html](Fiware/frontend/index.html) | **Macro twin** — full HUD (Handloom macro view) |
| [micro.html](Fiware/frontend/micro.html) | **Micro view** — breathing circuit + lung animation |
| [sandbox.html](Fiware/frontend/sandbox.html) | **Sandbox control panel** — what-if PEEP/FiO₂/TV |
| [intelligence.html](Fiware/frontend/intelligence.html) | **LSTM + PPO + replay** API console |

React app **Ventilator OS** also links via nav → **FIWARE Twin** (opens hub).

## Run (local)

```powershell
# 1) FastAPI co-pilot
cd "Major Project"
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# 2) Orion + MongoDB (optional but required for LIVE FIWARE pill)
cd ventilator-twin
docker compose up -d

# 3) Flask relay
cd Fiware\backend
pip install flask flask-cors requests
$env:FIWARE_ORION = "http://127.0.0.1:1026"
$env:DIGITAL_TWIN_API = "http://127.0.0.1:8000"
python relay.py

# 4) Dashboards
cd ..\frontend
python -m http.server 8080
# http://127.0.0.1:8080/hub.html
```

Set `FIWARE_API_VERSION=v2` on FastAPI when using classic Orion NGSI-v2 (default in relay).

## NGSI-v2 entity

- **Type:** `VentilatorTwin`
- **ID:** `VentilatorTwin:800000` (patient stay_id)
- Updated on each relay poll and on `/patient/{id}/recommend` via `services/fiware_adapter.py`

## Asset path

3D model from `medical-ventilator/source/Ventilator/` — served as:

- `Fiware/frontend/assets/medical-ventilator/model/Ventilator.dae`
- Textures under `assets/medical-ventilator/Textures/`

Also mirrored in `frontend/app/public/` for the React dashboard.
