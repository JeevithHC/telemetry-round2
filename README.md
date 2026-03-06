# 🚨 FleetSentinel — Real-Time Vehicle Fleet Fault Detection System

> AI-powered telemetry anomaly detection for a 100-vehicle Chennai fleet.  
> Detects 16 fault types across 7 vehicle classes in real time using Kalman filters, Z-score analysis, linear regression, and threshold-based alerting.

---

## 📌 What This Is

FleetSentinel is a full-stack fault detection pipeline built for a live vehicle telemetry hackathon challenge. It simulates a 100-vehicle fleet operating across Chennai, generates realistic physics-based sensor data, injects vehicle-type-aware faults, runs them through a multi-layer detection engine, and surfaces prioritized alerts.

The system is **not a dashboard that polls random numbers** — every alert is backed by a statistical or physics-based detection model. Oil pressure drops are detected by linear regression over a sliding window. Engine overheats are caught by both hard thresholds and Z-score anomaly detection. Tyre blowouts register within 1–2 ticks of the fault starting.

---

## 🏗️ Architecture

```
simulator.py
    │
    │  POST /api/telemetry  (1 payload/second/vehicle)
    ▼
server.py  (FastAPI)
    │
    │  bridge.process(payload)
    ▼
simulator_bridge.py
    │  ├── payload_to_snapshot()   field mapping + unit conversion
    │  ├── OilPressureDetector
    │  ├── TyrePressureDetector
    │  ├── TurboDetector
    │  ├── OverspeedDetector
    │  ├── DriverBehaviorDetector
    │  ├── FuelLeakDetector
    │  └── VehicleBridge.process()
    │           │
    │           ▼
    │    vehicle_safety_system_v3.py
    │           ├── EngineTempDetector
    │           ├── VibrationDetector
    │           ├── BatteryDetector
    │           ├── AirBrakeDetector
    │           ├── AxleOverloadDetector
    │           ├── BeltChainSlipDetector
    │           ├── BrakeWearDetector
    │           └── AlignmentDetector
    │
    │  GET /api/alerts  (frontend polls)
    ▼
fleetsentinal.html  (self-contained dashboard)
```

---

## 📁 File Reference

| File | Role | Can run standalone? |
|---|---|---|
| `simulator.py` | Generates telemetry for 100 vehicles, injects faults | ✅ Yes (just won't POST anywhere without server) |
| `vehicle_safety_system_v3.py` | Core detection engine — Kalman, Z-score, regression | ✅ Yes (pure library) |
| `simulator_bridge.py` | Maps simulator fields → detection engine input | ✅ Yes (has own demo runner) |
| `server.py` | FastAPI server — receives telemetry, serves alerts | ✅ Yes |
| `fleetsentinal.html` | Live dashboard — works completely standalone | ✅ Yes (open in browser) |

---

## 🚗 Fleet Composition

| Type | Count | Class | Special Faults |
|---|---|---|---|
| SCOOTY | 10 | Two-Wheeler | CHAIN_SLIP |
| BIKE | 10 | Two-Wheeler | CHAIN_SLIP |
| CAR | 20 | Car | BRAKE_WEAR |
| PICKUP | 10 | Car | BRAKE_WEAR, TURBO_FAILURE, OVERLOAD |
| VAN | 15 | Car | BRAKE_WEAR, TURBO_FAILURE, OVERLOAD |
| TRUCK | 20 | Heavy | AIR_BRAKE_LOSS, OVERLOAD, TYRE_BLOWOUT_HEAVY, TURBO_FAILURE |
| BUS | 15 | Heavy | AIR_BRAKE_LOSS, OVERLOAD, TYRE_BLOWOUT_HEAVY, TURBO_FAILURE |

GPS bounded to **Chennai**: `lat 12.92–13.18`, `lon 80.12–80.27`

---

## ⚡ Fault Catalogue

### CRITICAL — Immediate action required
| Fault | Triggered By | Field(s) Affected |
|---|---|---|
| ENGINE_OVERHEAT | engine_temp > 115–130% of max | `engine_temp`, `coolant_temp` |
| OIL_PRESSURE_DROP | Gradual ramp, 1–3 PSI/tick | `oil_pressure` |
| BATTERY_FAILURE | Battery drains + voltage collapses | `battery_level`, `alternator_voltage` |
| TYRE_BLOWOUT_FL | Front-left drops to <15% rated PSI | `tyre_pressure_fl` |
| TYRE_BLOWOUT_RR | Rear-right drops to <15% rated PSI | `tyre_pressure_rr` |
| TYRE_BLOWOUT_HEAVY | Both rears drop ~80 PSI (TRUCK/BUS) | `tyre_pressure_rl`, `tyre_pressure_rr` |
| AIR_BRAKE_LOSS | Brake pressure near 0 (TRUCK/BUS only) | `brake_pressure` |

### WARNING — Schedule inspection
| Fault | Triggered By | Field(s) Affected |
|---|---|---|
| COOLANT_LEAK | Gradual coolant temp ramp | `coolant_temp`, `engine_temp` |
| VIBRATION_SPIKE | Vibration > 110–155% of max | `engine_vibration` |
| FUEL_LEAK | Fuel drains at 3× normal rate | `fuel_level` |
| OVERSPEED | Speed > 130–145% of type max | `speed`, `rpm` |
| OVERLOAD | Load > 140% capacity (PICKUP/VAN/TRUCK/BUS) | `load_weight_pct`, rear tyres |
| TURBO_FAILURE | Boost collapses to 0, RPM surges | `turbo_boost`, `rpm` |
| CHAIN_SLIP | High RPM + speed drop (SCOOTY/BIKE) | `rpm`, `speed`, `engine_vibration` |
| BRAKE_WEAR | Persistently high brake pressure (CAR/PICKUP/VAN) | `brake_pressure` |

### INFO — Driver coaching
| Fault | Triggered By | Field(s) Affected |
|---|---|---|
| HARSH_DRIVER | Aggressive braking + acceleration | `brake_pressure`, `accelerator_pct`, `steering_angle` |
| GPS_DRIFT | Coordinates jump outside Chennai bounds | `latitude`, `longitude`, `gps_signal` |

---

## 🔬 Detection Methods

| Method | Used For |
|---|---|
| **Hard threshold** | Tyre blowout, overspeed, air brake loss, fuel critical |
| **Linear regression (sliding window)** | OIL_PRESSURE_DROP, COOLANT_LEAK gradual ramps |
| **Kalman filter** | Noise reduction on engine temp, vibration, battery |
| **Z-score anomaly detection** | VIBRATION_SPIKE, BATTERY_FAILURE, GPS_DRIFT |
| **Rate-of-change (delta/tick)** | FUEL_LEAK (drain rate vs. expected) |
| **Correlated multi-field** | ENGINE_OVERHEAT (temp + coolant together) |

---

## 📦 Installation

```bash
# Python dependencies
pip install fastapi uvicorn requests

# No other dependencies — all detection is pure Python stdlib + the above
```

**Required Python version:** 3.10+

---

## 🚀 Running the Full Pipeline

Place all files in the same directory:

```
project/
  server.py
  simulator.py
  simulator_bridge.py
  vehicle_safety_system_v3.py
  fleetsentinal.html
```

### Terminal 1 — Start the API server

```bash
python server.py
```

Wait for: `Uvicorn running on http://0.0.0.0:8000`

### Terminal 2 — Start the simulator

```bash
python simulator.py
```

You should see telemetry streaming:
```
[CAR-007] 0.01h | 45.3km/h | 2341rpm | eng=72.4C | fuel=98.2% | health=94
[TRUCK-003] 0.01h | 62.1km/h | 1580rpm | eng=78.1C | fuel=99.1% | health=91 | FAULT=OVERLOAD
```

### Browser — Open the dashboard

Double-click `fleetsentinal.html` — opens locally, no web server needed.

### Optional — Check live alerts via curl

```bash
curl http://localhost:8000/api/alerts
curl http://localhost:8000/api/stats
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/telemetry` | Receives one simulator tick payload |
| `GET` | `/api/alerts?limit=100` | Returns latest N alerts |
| `GET` | `/api/stats` | Fleet summary — total, critical, warning counts |

### Telemetry Payload Shape (simulator → server)

```json
{
  "vehicle_id":       "TRUCK-007",
  "vehicle_type":     "TRUCK",
  "timestamp":        "2026-03-06T10:00:00+00:00",
  "data_type":        "fault",
  "speed":            112.5,
  "rpm":              2800,
  "engine_temp":      132.4,
  "coolant_temp":     118.2,
  "oil_pressure":     14.3,
  "engine_vibration": 5.8,
  "tyre_pressure_fl": 98.2,
  "tyre_pressure_rr": 4.1,
  "fuel_level":       62.4,
  "battery_level":    88.1,
  "alternator_voltage": 14.2,
  "brake_pressure":   22.0,
  "load_weight_pct":  155.0,
  "latitude":         13.0451,
  "longitude":        80.2101,
  "active_fault":     "TYRE_BLOWOUT_RR",
  "fault_severity":   "CRITICAL",
  "health_score":     28.0,
  "driver_safety_score": 85.0,
  "maintenance_required": true
}
```

### Alert Response Shape (server → frontend)

```json
{
  "vehicle_id":   "TRUCK-007",
  "vehicle_type": "TRUCK",
  "module":       "TyrePressureDetector",
  "severity":     "CRITICAL",
  "layer":        "CRITICAL",
  "score":        94.2,
  "message":      "Tyre RR: 4.1 PSI = 4% of nominal (110 PSI).",
  "value":        4.1,
  "timestamp":    "2026-03-06T10:00:00+00:00",
  "active_fault": "TYRE_BLOWOUT_RR"
}
```

---

## 🧠 Sensor Field → Detector Mapping

| Simulator Field | Detector | Alert Layers |
|---|---|---|
| `engine_temp` / `coolant_temp` | EngineTempDetector | CRITICAL / PREDICTIVE / ANOMALY |
| `oil_pressure` | OilPressureDetector | CRITICAL / PREDICTIVE |
| `alternator_voltage` | BatteryDetector | CRITICAL / PREDICTIVE |
| `tyre_pressure_fl/fr/rl/rr` | TyrePressureDetector | CRITICAL / PREDICTIVE |
| `brake_pressure` (TRUCK/BUS) | AirBrakeDetector | CRITICAL / ANOMALY |
| `engine_vibration` | VibrationDetector | CRITICAL / ANOMALY / PREDICTIVE |
| `load_weight_pct` | AxleOverloadDetector | CRITICAL |
| `rpm` + `speed` (2-wheeler) | BeltChainSlipDetector | ANOMALY |
| `brake_pressure` + decel | BrakeWearDetector | PREDICTIVE |
| `turbo_boost` | TurboDetector | ANOMALY |
| `speed` vs `max_speed` | OverspeedDetector | CRITICAL |
| `harsh_braking` / `harsh_acceleration` / `steering_angle` | DriverBehaviorDetector | ANOMALY |
| `fuel_level` drain rate | FuelLeakDetector | ANOMALY |
| GPS `lat`/`lon` drift | AlignmentDetector | PREDICTIVE / ANOMALY |

---

## 📊 Alert Severity Scoring

Alerts are scored 0–100 using a composite formula:

```
score = threshold_factor (0–60 pts)
      + gate_count_factor (0–30 pts)   ← sustained events score higher
      + cross_factor      (0–10 pts)   ← hard threshold breach bonus
      + z_score_factor    (0–10 pts)   ← statistical unusualness bonus
```

| Score Range | Band | Action |
|---|---|---|
| 71–100 | 🚨 CRITICAL | Immediate stop / intervention |
| 31–70 | ⚠️ WARNING | Schedule inspection within 24h |
| 0–30 | 📋 LOG | Record for maintenance history |

---

## 👥 Team

| Member | Responsibility |
|---|---|
| Backend | API server, telemetry ingestion, alert routing, storage |
| Safety Engine | Detection algorithms, Kalman filters, Z-score models |
| Bridge | Simulator integration, field mapping, unit conversion |
| Frontend | Real-time dashboard, alert display, fleet visualization |

---

## 🔭 Future Scope

- **Persistent storage** — SQLite/PostgreSQL for telemetry history and alert replay
- **ML predictive maintenance** — train models on historical fault sequences
- **OBD-II live integration** — replace simulator with real vehicle hardware feeds
- **Driver mobile alerts** — push CRITICAL notifications directly to driver devices
- **Live GPS heatmap** — fleet-wide anomaly overlay on Chennai road map
- **Auto-escalation** — CRITICAL faults trigger automatic workshop booking
- **Driver scoring over time** — gamified improvement tracking per driver ID

---

## ⚠️ Known Limitations

- Telemetry is currently stored **in-memory only** (last 500 alerts) — restarting the server clears all data
- The frontend dashboard (`fleetsentinal.html`) runs its **own internal simulation** and is not yet wired to the live API server
- No authentication on API endpoints — not production-ready

---

*Built for the Vehicle Telemetry Fault Detection hackathon challenge. Chennai fleet. 100 vehicles. 35+ sensor fields per second.*
