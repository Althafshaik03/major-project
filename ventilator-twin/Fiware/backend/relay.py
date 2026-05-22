"""
Ventilator Digital Twin — Flask Relay (Handloom-style FIWARE middleware)

Bridges:
  FastAPI ventilator backend (:8000)  →  NGSI-v2 Orion (:1026)  →  Three.js dashboards

Port: 5050
"""

from __future__ import annotations

import os
import time
import math
import random
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DIGITAL_TWIN_API = os.getenv("DIGITAL_TWIN_API", "http://127.0.0.1:8000")
FIWARE_ORION = os.getenv("FIWARE_ORION", "http://127.0.0.1:1026").rstrip("/")
FIWARE_SERVICE = os.getenv("FIWARE_SERVICE", "openiot")
FIWARE_SERVICE_PATH = os.getenv("FIWARE_SERVICE_PATH", "/")
FIWARE_ENABLED = os.getenv("FIWARE_ENABLED", "true").lower() not in ("0", "false", "no")
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() not in ("0", "false", "no")
DEFAULT_PATIENT_ID = int(os.getenv("DEFAULT_PATIENT_ID", "800000"))
POLL_MS = int(os.getenv("RELAY_POLL_MS", "500"))

event_log: deque = deque(maxlen=120)
current_state: Dict[str, Any] = {}
recommendation_cache: Dict[str, Any] = {}


def _log(kind: str, message: str) -> None:
    entry = {
        "ts": datetime.now().strftime("%I:%M:%S %p"),
        "kind": kind,
        "message": message,
    }
    event_log.appendleft(entry)
    print(f"[{kind}] {message}")


def _fiware_headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Fiware-Service": FIWARE_SERVICE,
        "Fiware-ServicePath": FIWARE_SERVICE_PATH,
    }


def _entity_id(patient_id: int) -> str:
    return f"VentilatorTwin:{patient_id}"


def orion_health() -> Dict[str, Any]:
    if not FIWARE_ENABLED:
        return {"enabled": False, "reachable": False, "detail": "FIWARE_ENABLED=false"}
    try:
        res = requests.get(f"{FIWARE_ORION}/version", timeout=2)
        return {
            "enabled": True,
            "reachable": res.status_code == 200,
            "version": res.json() if res.status_code == 200 else None,
            "url": FIWARE_ORION,
        }
    except Exception as exc:
        return {"enabled": True, "reachable": False, "detail": str(exc)}


def publish_to_orion(patient_id: int, attributes: Dict[str, Any]) -> bool:
    if not FIWARE_ENABLED:
        return False

    entity: Dict[str, Any] = {
        "id": _entity_id(patient_id),
        "type": "VentilatorTwin",
    }
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (list, dict)):
            entity[key] = {"type": "Text", "value": json_dumps_safe(value)}
        elif isinstance(value, bool):
            entity[key] = {"type": "Boolean", "value": value}
        elif isinstance(value, int) and not isinstance(value, bool):
            entity[key] = {"type": "Integer", "value": value}
        elif isinstance(value, float):
            entity[key] = {"type": "Float", "value": value}
        elif isinstance(value, str):
            entity[key] = {"type": "Text", "value": value}
        else:
            try:
                entity[key] = {"type": "Float", "value": float(value)}
            except (TypeError, ValueError):
                entity[key] = {"type": "Text", "value": str(value)}

    try:
        res = requests.post(
            f"{FIWARE_ORION}/v2/entities?options=upsert",
            headers=_fiware_headers(),
            json=entity,
            timeout=3,
        )
        if res.status_code not in (200, 201, 204):
            _log("FIWARE", f"publish failed {res.status_code}: {res.text[:120]}")
            return False
        return True
    except Exception as exc:
        _log("FIWARE", f"publish error: {exc}")
        return False


def json_dumps_safe(value: Any) -> str:
    import json
    return json.dumps(value)


def read_from_orion(patient_id: int) -> Optional[Dict[str, Any]]:
    if not FIWARE_ENABLED:
        return None
    try:
        res = requests.get(
            f"{FIWARE_ORION}/v2/entities/{_entity_id(patient_id)}",
            headers=_fiware_headers(),
            timeout=2,
        )
        if res.status_code != 200:
            return None
        raw = res.json()
        flat: Dict[str, Any] = {"patient_id": patient_id}
        for key, val in raw.items():
            if key in ("id", "type"):
                continue
            if isinstance(val, dict) and "value" in val:
                flat[key] = val["value"]
            else:
                flat[key] = val
        return flat
    except Exception:
        return None


def fetch_patient_state(patient_id: int) -> Optional[Dict[str, Any]]:
    try:
        tick = requests.post(f"{DIGITAL_TWIN_API}/patient/{patient_id}/tick", timeout=3)
        latest = {}
        if tick.status_code == 200:
            latest = tick.json().get("latest_record") or {}
        if not latest:
            hist = requests.get(f"{DIGITAL_TWIN_API}/patient/{patient_id}/history", timeout=3)
            hist.raise_for_status()
            rows = hist.json().get("history", [])
            latest = rows[-1] if rows else {}
        if not latest:
            return None

        spo2 = float(latest.get("SpO2", 97.0) or 97.0)
        pressure = float(latest.get("PlateauPressure", latest.get("Pressure", 15.0)) or 15.0)
        return {
            "patient_id": patient_id,
            "peep": float(latest.get("PEEP", 5.0) or 5.0),
            "fio2": float(latest.get("FiO2", 40.0) or 40.0),
            "tidal_vol": float(latest.get("TidalVol", latest.get("TidalVolume", 450.0)) or 450.0),
            "resp_rate": float(latest.get("RespRate", 12.0) or 12.0),
            "pressure": pressure,
            "spo2": spo2,
            "spo2_predicted": spo2,
            "hr": float(latest.get("HR", 80.0) or 80.0),
            "map": float(latest.get("MAP", 75.0) or 75.0),
            "temperature": float(latest.get("Temp", latest.get("Temperature", 36.8)) or 36.8),
            "status": "critical" if spo2 < 88 else "warning" if spo2 < 93 else "stable",
            "quality_grade": "A" if spo2 >= 95 else "B" if spo2 >= 92 else "C",
            "defect_rate": max(0.0, (100.0 - spo2) * 0.8),
            "anomaly_score": max(0.0, min(1.0, (93.0 - spo2) / 10.0)),
            "cycles": int(time.time()) % 1000000,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "fastapi",
        }
    except Exception as exc:
        _log("RELAY", f"API fetch failed: {exc}")
        return None


def fetch_recommendation(patient_id: int) -> Optional[Dict[str, Any]]:
    try:
        hist = requests.get(f"{DIGITAL_TWIN_API}/patient/{patient_id}/history", timeout=3)
        hist.raise_for_status()
        rows = hist.json().get("history", [])[-96:]
        if not rows:
            return None
        latest = rows[-1]
        res = requests.post(
            f"{DIGITAL_TWIN_API}/patient/{patient_id}/recommend",
            json={**latest, "history": rows},
            timeout=8,
        )
        if res.status_code == 200:
            return res.json()
    except Exception as exc:
        _log("PRED", f"recommend failed: {exc}")
    return None


def generate_mock_state(patient_id: int = DEFAULT_PATIENT_ID) -> Dict[str, Any]:
    t = time.time() % 60 / 60.0
    breath = math.sin(t * 2 * math.pi)
    spo2 = 96.5 + random.gauss(0, 0.6)
    return {
        "patient_id": patient_id,
        "peep": 5.0 + random.gauss(0, 0.3),
        "fio2": 40.0 + random.gauss(0, 2),
        "tidal_vol": 450.0 + breath * 40,
        "resp_rate": 12 + random.randint(-1, 1),
        "pressure": 15.0 + breath * 6,
        "spo2": spo2,
        "spo2_predicted": spo2 + random.gauss(0, 0.4),
        "hr": 80 + random.gauss(0, 3),
        "map": 75 + random.gauss(0, 2),
        "temperature": 36.8 + random.gauss(0, 0.15),
        "status": "stable" if spo2 > 92 else "warning",
        "quality_grade": "A" if spo2 > 94 else "B",
        "defect_rate": max(0.0, (100 - spo2) * 0.7),
        "anomaly_score": max(0.0, min(1.0, (93 - spo2) / 8)),
        "cycles": int(time.time()) % 1000000,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "demo",
    }


def sync_state(patient_id: int) -> Dict[str, Any]:
    global current_state, recommendation_cache

    api_state = fetch_patient_state(patient_id)
    if api_state:
        current_state = api_state
        _log("TELEMETRY", f"patient={patient_id} spo2={api_state['spo2']:.1f} peep={api_state['peep']:.1f}")
    elif DEMO_MODE:
        current_state = generate_mock_state(patient_id)
        _log("DEMO", f"mock telemetry patient={patient_id}")

    rec = fetch_recommendation(patient_id)
    if rec:
        recommendation_cache = rec
        twin = rec.get("twin_simulation") or {}
        current_state["spo2_predicted"] = rec.get("pred_next_spo2", current_state.get("spo2"))
        current_state["hypoxia_prob"] = rec.get("hypoxia_prob")
        current_state["alert_level"] = rec.get("alert_level")
        current_state["proposed_peep"] = rec.get("proposed", {}).get("PEEP")
        current_state["proposed_fio2"] = rec.get("proposed", {}).get("FiO2")
        current_state["proposed_tv"] = rec.get("proposed", {}).get("TidalVol")
        current_state["twin_mean_spo2"] = twin.get("mean_spo2")
        current_state["twin_risk_flag"] = twin.get("risk_flag")
        _log("PRED", f"hypoxia={rec.get('hypoxia_prob', 0):.2f} alert={rec.get('alert_level')}")

    skip_keys = {"recommendation", "source", "timestamp"}
    attrs = {k: v for k, v in current_state.items() if k not in skip_keys}
    if recommendation_cache:
        attrs["alertLevel"] = recommendation_cache.get("alert_level")
        attrs["predNextSpO2"] = recommendation_cache.get("pred_next_spo2")
        attrs["hypoxiaProb"] = recommendation_cache.get("hypoxia_prob")
    try:
        if publish_to_orion(patient_id, attrs):
            _log("FIWARE", f"NGSI-v2 upsert {_entity_id(patient_id)}")
    except Exception as exc:
        _log("FIWARE", f"publish skipped: {exc}")
    return current_state


# ─── REST API ───────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    api_ok = False
    try:
        api_ok = requests.get(f"{DIGITAL_TWIN_API}/health", timeout=2).status_code == 200
    except Exception:
        pass
    orion = orion_health()
    return jsonify({
        "status": "ok",
        "digital_twin_api": "ok" if api_ok else "offline",
        "demo_mode": DEMO_MODE,
        "fiware": orion,
        "default_patient_id": DEFAULT_PATIENT_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/fiware/status")
def fiware_status():
    orion = orion_health()
    entity = read_from_orion(DEFAULT_PATIENT_ID)
    return jsonify({
        "orion": orion,
        "entity_id": _entity_id(DEFAULT_PATIENT_ID),
        "entity_sample": entity,
        "service": FIWARE_SERVICE,
        "service_path": FIWARE_SERVICE_PATH,
    })


@app.route("/events")
def events():
    return jsonify({"events": list(event_log)})


@app.route("/twin")
def get_twin():
    patient_id = request.args.get("patient_id", DEFAULT_PATIENT_ID, type=int)
    state = sync_state(patient_id)
    orion = read_from_orion(patient_id)
    return jsonify({
        **state,
        "recommendation": recommendation_cache,
        "fiware_online": orion_health().get("reachable", False),
        "orion_entity": orion,
    })


@app.route("/twin/history")
def get_twin_history():
    patient_id = request.args.get("patient_id", DEFAULT_PATIENT_ID, type=int)
    limit = request.args.get("limit", 64, type=int)
    try:
        res = requests.get(f"{DIGITAL_TWIN_API}/patient/{patient_id}/history", timeout=3)
        if res.status_code == 200:
            rows = res.json().get("history", [])[-limit:]
            return jsonify({"history": rows, "source": "fastapi"})
    except Exception:
        pass
    history = []
    for i in range(limit):
        history.append(generate_mock_state(patient_id))
    return jsonify({"history": history, "source": "demo"})


@app.route("/twin/recommend")
def get_recommendation_route():
    patient_id = request.args.get("patient_id", DEFAULT_PATIENT_ID, type=int)
    rec = fetch_recommendation(patient_id)
    if rec:
        return jsonify(rec)
    return jsonify({
        "alert_level": "STABLE",
        "pred_next_spo2": current_state.get("spo2_predicted", 95),
        "hypoxia_prob": 0.05,
        "proposed": {"PEEP": 5, "FiO2": 40, "TidalVol": 450},
        "rationale": "Demo recommendation — start FastAPI on :8000 for live PPO+LSTM",
    })


@app.route("/twin/apply", methods=["POST"])
def apply_parameters():
    data = request.json or {}
    patient_id = data.get("patient_id", DEFAULT_PATIENT_ID)
    _log("APPLY", f"queued settings for patient {patient_id}")
    publish_to_orion(
        int(patient_id),
        {
            "proposedPEEP": data.get("peep"),
            "proposedFiO2": data.get("fio2"),
            "proposedTidalVol": data.get("tidal_vol"),
            "eventSource": "sandbox-apply",
        },
    )
    return jsonify({"success": True, "patient_id": patient_id, "queued": data})


@app.route("/sandbox")
def sandbox_state():
    return jsonify({
        "mode": "sandbox",
        "is_isolated": True,
        "patient_id": DEFAULT_PATIENT_ID,
        "last_sync": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/sandbox/simulate", methods=["POST"])
def sandbox_simulate():
    data = request.json or {}
    patient_id = int(data.get("patient_id", DEFAULT_PATIENT_ID))
    try:
        hist = requests.get(f"{DIGITAL_TWIN_API}/patient/{patient_id}/history", timeout=3)
        rows = hist.json().get("history", [])[-32:] if hist.status_code == 200 else []
        latest = rows[-1] if rows else {}
        body = {
            "stay_id": patient_id,
            "history": rows,
            "current_spo2": data.get("current_spo2", latest.get("SpO2")),
            "proposed": data.get("proposed") or {
                "PEEP": data.get("peep", 5),
                "FiO2": data.get("fio2", 40),
                "TidalVol": data.get("tidal_vol", 450),
            },
            "steps": data.get("steps", 8),
            "noise_scale": data.get("noise_scale", 0),
        }
        res = requests.post(f"{DIGITAL_TWIN_API}/twin/replay", json=body, timeout=8)
        if res.status_code == 200:
            result = res.json()
            publish_to_orion(patient_id, {"eventSource": "sandbox-replay", "replay": result})
            _log("SANDBOX", f"replay mean_spo2={result.get('result', {}).get('mean_spo2')}")
            return jsonify(result)
    except Exception as exc:
        _log("SANDBOX", f"replay failed: {exc}")

    mock_traj = [data.get("current_spo2", 95) - i * 0.2 for i in range(5)]
    return jsonify({
        "mode": "demo",
        "result": {
            "trajectory": mock_traj,
            "mean_spo2": sum(mock_traj) / len(mock_traj),
            "delta_spo2": mock_traj[-1] - mock_traj[0],
            "uncertainty": 1.2,
            "risk_flag": mock_traj[-1] < 90,
        },
    })


def background_updater():
    while True:
        try:
            sync_state(DEFAULT_PATIENT_ID)
        except Exception as exc:
            _log("BG", str(exc))
        time.sleep(max(0.2, POLL_MS / 1000.0))


threading.Thread(target=background_updater, daemon=True).start()


if __name__ == "__main__":
    print("=" * 60)
    print("Ventilator FIWARE Relay")
    print(f"  FastAPI : {DIGITAL_TWIN_API}")
    print(f"  Orion   : {FIWARE_ORION} (enabled={FIWARE_ENABLED})")
    print(f"  Demo    : {DEMO_MODE}")
    print("  http://localhost:5050")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
