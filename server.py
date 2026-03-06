"""
server.py — FleetSentinel API Server
======================================
Receives telemetry from simulator.py, runs it through
SimulatorBridge (detection engine), and serves alerts
to the frontend via polling.

Install once:
    pip install fastapi uvicorn

Run:
    python server.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from collections import deque
import uvicorn

from simulator_bridge import SimulatorBridge

app    = FastAPI()
bridge = SimulatorBridge()

# Store last 500 alerts in memory — frontend polls this
alert_store: deque = deque(maxlen=500)

# Allow the HTML file opened locally to call this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/telemetry", status_code=201)
async def receive_telemetry(payload: dict):
    """Simulator posts here every second per vehicle."""
    score, alerts = bridge.process(payload)
    for a in alerts:
        alert_store.appendleft({
            "vehicle_id":   payload.get("vehicle_id"),
            "vehicle_type": payload.get("vehicle_type"),
            "module":       a.module,
            "severity":     a.severity.value,      # "CRITICAL" / "WARNING" / "LOG"
            "layer":        a.layer.value,          # "CRITICAL" / "ANOMALY" / "PREDICTIVE"
            "score":        round(a.score, 1),
            "message":      a.message,
            "value":        round(a.value, 2) if a.value is not None else None,
            "timestamp":    payload.get("timestamp"),
            "active_fault": payload.get("active_fault", ""),
        })
    return {"status": "ok", "score": round(score, 1), "alerts": len(alerts)}


@app.get("/api/alerts")
async def get_alerts(limit: int = 100):
    """Frontend polls this to show live alerts."""
    return list(alert_store)[:limit]


@app.get("/api/stats")
async def get_stats():
    """Quick fleet summary for dashboard counters."""
    alerts = list(alert_store)
    return {
        "total_alerts":    len(alerts),
        "critical":        sum(1 for a in alerts if "CRITICAL" in a["severity"]),
        "warning":         sum(1 for a in alerts if "WARNING"  in a["severity"]),
        "vehicles_seen":   bridge.vehicle_count(),
    }


if __name__ == "__main__":
    print("=" * 50)
    print("  FleetSentinel API Server")
    print("  POST /api/telemetry  ← simulator")
    print("  GET  /api/alerts     ← frontend")
    print("  GET  /api/stats      ← frontend")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
