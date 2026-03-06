"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         UNIFIED PREDICTIVE VEHICLE SAFETY SYSTEM  v3.0                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DETECTION LAYERS                                                           ║
║  ① REACTIVE   — Hard threshold breach       (it IS broken)                 ║
║  ② PROACTIVE  — Trend / regression slope    (it IS wearing out)            ║
║  ③ PREDICTIVE — Z-Score anomaly detection   (it WILL break soon)           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  VEHICLE CLASSES  (select at runtime)                                       ║
║  🚛 HEAVY   — Bus / Truck / Van     (J1939 / air brake / DPF)              ║
║  🚗 CAR     — Personal commuter     (OBD-II / emissions)                   ║
║  🛵 TWO_WHEELER — Scooty / Bike     (IMU-heavy / derived logic)            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SMART VALIDATION (Anti-Alert-Fatigue)                                      ║
║  • Persistence Check   — signal must last > N frames                       ║
║  • Cross-Sensor Check  — corroborate with a second signal                  ║
║  • Context Filters     — cold start / mountain / rough road / cold weather  ║
║  • Severity Score 0-100 → LOG / WARNING / CRITICAL                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  FAULT DETECTORS                                                            ║
║  SHARED   ①Engine Temp ②Battery ③Alignment ④Vibration                     ║
║  HEAVY    ⑤Air Brake   ⑥DPF Regen ⑦Axle Overload ⑧Cooling Inefficiency  ║
║  CAR      ⑨Misfire     (+ shared ②③④)                                    ║
║  TWO_WHEEL ⑩Tip-over  ⑪Belt/Chain Slip  ⑫Brake Wear                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENUMERATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VehicleClass(Enum):
    HEAVY      = "🚛 HEAVY"
    CAR        = "🚗 CAR"
    TWO_WHEELER = "🛵 TWO-WHEELER"


class AlertLayer(Enum):
    PREDICTIVE = "🔵 PREDICTIVE"
    ANOMALY    = "🟡 ANOMALY"
    CRITICAL   = "🔴 CRITICAL"


class FaultCategory(Enum):
    DIRECT  = "DIRECT"    # sensor itself says "broken"
    DERIVED = "DERIVED"   # multi-sensor logic


class SeverityBand(Enum):
    LOG      = "📋 LOG"       # score  0–30  — maintenance record only
    WARNING  = "⚠️  WARNING"  # score 31–70  — yellow lamp / suggest check
    CRITICAL = "🚨 CRITICAL"  # score 71–100 — red lamp / buzzer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ALERT DATACLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Alert:
    module:       str
    layer:        AlertLayer
    category:     FaultCategory
    severity:     SeverityBand
    score:        float           # 0–100 composite severity score
    message:      str
    value:        float
    threshold:    float
    eta_frames:   Optional[float] = None

    def __str__(self) -> str:
        eta = f"  → ETA to limit   : {self.eta_frames:.0f} frames\n" if self.eta_frames else ""
        return (
            f"\n  {self.layer.value}  [{self.module}]  "
            f"[{self.category.value}]  {self.severity.value}  score={self.score:.0f}\n"
            f"  → {self.message}\n"
            f"{eta}"
            f"  → Measured={self.value:.3f}  Threshold={self.threshold:.3f}"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UNIFIED VEHICLE SNAPSHOT  (superset of all vehicle classes)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class VehicleSnapshot:
    # ── OBD-II / J1939 shared ────────────────────
    rpm:             float = 800.0    # Engine RPM
    speed_kmh:       float = 0.0
    engine_load:     float = 30.0    # % (0–100)
    coolant_temp:    float = 85.0    # °C
    battery_voltage: float = 12.6    # V
    alternator_v:    float = 14.2    # V  (alternator output)
    oil_temp:        float = 85.0    # °C (cross-validation for coolant anomaly)
    fuel_trim:       float = 0.0     # % (short-term fuel trim)

    # ── IMU ──────────────────────────────────────
    lateral_accel:     float = 0.0   # g
    longitudinal_accel: float = 0.0  # g
    vertical_accel:    float = 1.0   # g  (1.0 = stationary, upright)
    lean_angle_deg:    float = 0.0   # degrees from vertical (two-wheelers)
    vibration_rms:     float = 0.08  # g  RMS
    incline_deg:       float = 0.0   # road grade in degrees (from IMU/GPS)

    # ── GPS / Steering ────────────────────────────
    steering_angle:    float = 0.0   # degrees
    gps_heading_deg:   float = 0.0
    lateral_drift_m:   float = 0.0   # cumulative metres
    elevation_m:       float = 0.0   # metres ASL  (for mountain context)

    # ── Wheel speeds (all 4 for car; rear-only for 2-wheeler) ────────
    wheel_speed_fl:  float = 0.0    # km/h front-left
    wheel_speed_fr:  float = 0.0    # km/h front-right
    wheel_speed_rl:  float = 0.0    # km/h rear-left
    wheel_speed_rr:  float = 0.0    # km/h rear-right
    wheel_speed_rear: float = 0.0   # km/h rear (two-wheeler)

    # ── Crankshaft / Ignition ─────────────────────
    crank_interval_ms: float = 0.0  # ms between TDC pulses (misfire detection)
    o2_sensor_voltage: float = 0.45 # V  (lambda ~1.0)

    # ── Brake ─────────────────────────────────────
    brake_lever_pressed: bool  = False
    brake_duration_s:    float = 0.0   # how long brake held this decel event
    decel_rate_ms2:      float = 0.0   # m/s² achieved during braking

    # ── Heavy-vehicle specific (J1939 / air system) ───────────────────
    brake_air_pressure_primary:   float = 120.0  # PSI
    brake_air_pressure_secondary: float = 120.0  # PSI
    dpf_backpressure_kpa:         float = 5.0    # kPa  (diesel particulate filter)
    exhaust_temp_c:               float = 400.0  # °C
    scr_status_ok:                bool  = True   # Selective Catalytic Reduction OK
    air_susp_pressure_psi:        float = 100.0  # PSI  (per axle)
    fan_speed_pct:                float = 50.0   # % of max fan RPM

    # ── Ambient ────────────────────────────────────
    ambient_temp:  float = 30.0   # °C
    is_cranking:   bool  = False

    # ── Meta ──────────────────────────────────────
    timestamp:     float = field(default_factory=time.time)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIGNAL PROCESSING PRIMITIVES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class KalmanFilter1D:
    def __init__(self, Q: float = 0.01, R: float = 0.5):
        self.Q = Q; self.R = R
        self.x: Optional[float] = None
        self.P = 1.0

    def update(self, z: float) -> float:
        if self.x is None:
            self.x = z; return z
        P_ = self.P + self.Q
        K  = P_ / (P_ + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * P_
        return self.x


class MovingWindow:
    def __init__(self, maxlen: int = 30):
        self.buf = deque(maxlen=maxlen)

    def push(self, v: float) -> "MovingWindow":
        self.buf.append(v); return self

    def mean(self) -> float:
        return sum(self.buf) / len(self.buf) if self.buf else 0.0

    def std(self) -> float:
        if len(self.buf) < 2: return 0.0
        m = self.mean()
        return math.sqrt(sum((x - m) ** 2 for x in self.buf) / len(self.buf))

    def z_score(self, v: float) -> float:
        s = self.std()
        return (v - self.mean()) / s if s > 1e-9 else 0.0

    def linear_regression(self) -> tuple[float, float]:
        n = len(self.buf)
        if n < 3: return 0.0, self.mean()
        xs = list(range(n)); ys = list(self.buf)
        mx = sum(xs) / n;    my = sum(ys) / n
        num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
        den = sum((xs[i]-mx)**2 for i in range(n))
        s = num / den if den > 1e-9 else 0.0
        return s, my - s * mx

    def eta_to(self, target: float) -> Optional[float]:
        slope, _ = self.linear_regression()
        if slope <= 0: return None
        cur = self.mean()
        if cur >= target: return 0.0
        return (target - cur) / slope

    def __len__(self) -> int: return len(self.buf)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SMART VALIDATION ENGINE
#  Wraps every raw signal with persistence, cross-sensor, context, and scoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PersistenceGate:
    """Signal must be True for at least `required` consecutive frames."""
    def __init__(self, required: int = 5):
        self.required = required
        self.count    = 0

    def update(self, condition: bool) -> bool:
        if condition:
            self.count += 1
        else:
            self.count = max(0, self.count - 1)
        return self.count >= self.required


class ContextFilter:
    """Returns True when context is 'safe to run diagnostics'."""

    @staticmethod
    def cold_start_ok(snap: VehicleSnapshot) -> bool:
        """Only run engine diagnostics after warm-up (coolant > 70°C)."""
        return snap.coolant_temp >= 70.0

    @staticmethod
    def not_mountain_driving(snap: VehicleSnapshot) -> bool:
        """Suppress cooling alerts when on steep incline (high load is expected)."""
        return abs(snap.incline_deg) < 8.0 or snap.speed_kmh > 5.0

    @staticmethod
    def not_rough_road(vibration_rms: float, moving_avg: float) -> bool:
        """True if spike is sustained (not a single pothole)."""
        return True   # Persistence gate handles timing; this flag can add GPS context

    @staticmethod
    def battery_ambient_correction(threshold_v: float, ambient_c: float) -> float:
        """Lower acceptable crank voltage in cold weather (chemistry slows down)."""
        if ambient_c < 0:
            return threshold_v - 0.4
        if ambient_c < 10:
            return threshold_v - 0.2
        return threshold_v


def severity_score(
    base: float,                   # raw measured deviation (0–1 normalised)
    persistence_frames: int,       # how long the condition lasted
    cross_sensor_match: bool,      # does a 2nd sensor agree?
    z_score: float = 0.0           # statistical unusualness
) -> float:
    """
    Three-step scoring as specified:
      Step 1  Persistence weight  — longer = more serious
      Step 2  Cross-sensor bonus  — corroboration raises confidence
      Step 3  Z-score factor      — statistical anomalousness
    Returns 0–100 composite score.
    """
    p_weight  = min(1.0, persistence_frames / 20.0) * 40    # max 40 pts
    base_pts  = min(1.0, base) * 30                          # max 30 pts
    cross_pts = 20 if cross_sensor_match else 0              # 20 pts if corroborated
    z_pts     = min(1.0, abs(z_score) / 4.0) * 10           # max 10 pts
    return min(100.0, p_weight + base_pts + cross_pts + z_pts)


def band(score: float) -> SeverityBand:
    if score >= 71: return SeverityBand.CRITICAL
    if score >= 31: return SeverityBand.WARNING
    return SeverityBand.LOG


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ══ SHARED DETECTORS (used by all vehicle classes) ══
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EngineTempDetector:
    CRITICAL = 105.0; SLOPE_THR = 0.3; MODEL_MARGIN = 3.5
    BASE = 75.0; LC = 0.18; AC = 0.25

    def __init__(self):
        self.kf = KalmanFilter1D(0.02, 1.5)
        self.win = MovingWindow(20); self.load_win = MovingWindow(15)
        self.gate = PersistenceGate(5)

    def _model(self, load, amb): return self.BASE + self.LC*load + self.AC*max(0,amb-20)

    def analyze(self, snap: VehicleSnapshot) -> list[Alert]:
        if not ContextFilter.cold_start_ok(snap): return []
        if not ContextFilter.not_mountain_driving(snap): return []
        alerts = []
        t = self.kf.update(snap.coolant_temp)
        self.win.push(t); avg_load = self.load_win.push(snap.engine_load).mean()

        if t >= self.CRITICAL:
            sc = severity_score(1.0, 20, snap.oil_temp > 100, self.win.z_score(t))
            alerts.append(Alert("ENGINE TEMP", AlertLayer.CRITICAL, FaultCategory.DERIVED,
                band(sc), sc, "Coolant temp at hardware limit. Stop safely NOW.",
                t, self.CRITICAL)); return alerts

        if len(self.win) >= 5:
            slope, _ = self.win.linear_regression()
            if self.gate.update(slope >= self.SLOPE_THR):
                eta = (self.CRITICAL - t) / slope if slope > 0 else None
                cross = snap.oil_temp > 95
                sc = severity_score(slope/1.0, self.gate.count, cross, self.win.z_score(t))
                alerts.append(Alert("ENGINE TEMP", AlertLayer.PREDICTIVE, FaultCategory.DERIVED,
                    band(sc), sc,
                    f"Temp rising {slope:.2f}°C/frame. Oil-temp corroboration: {'YES' if cross else 'NO'}.",
                    t, self.CRITICAL, eta))

        exp = self._model(avg_load, snap.ambient_temp)
        delta = t - exp; z = self.win.z_score(t)
        if delta >= self.MODEL_MARGIN:
            cross = snap.oil_temp > exp + 2
            sc = severity_score(delta/10, self.gate.count, cross, z)
            alerts.append(Alert("ENGINE TEMP", AlertLayer.ANOMALY, FaultCategory.DERIVED,
                band(sc), sc,
                f"Temp {delta:.1f}°C above thermo-model ({exp:.1f}°C expected). Z={z:.2f}.",
                t, exp + self.MODEL_MARGIN))
        return alerts


class BatteryDetector:
    HEALTHY = 9.6; WARN = 9.0; CRITICAL = 7.5

    def __init__(self):
        self.kf = KalmanFilter1D(0.005, 0.1)
        self.hist = MovingWindow(20); self.gate = PersistenceGate(3)

    def _health(self, v):
        if v >= 9.6: return 70 + (v-9.6)/(12.6-9.6)*30
        if v >= 9.0: return 30 + (v-9.0)/(9.6-9.0)*40
        if v >= 7.5: return     (v-7.5)/(9.0-7.5)*30
        return 0.0

    def analyze(self, snap: VehicleSnapshot) -> list[Alert]:
        alerts = []
        v = self.kf.update(snap.battery_voltage)
        self.hist.push(v)

        # Ambient-corrected threshold
        warn_thr = ContextFilter.battery_ambient_correction(self.WARN, snap.ambient_temp)

        if snap.is_cranking:
            if v <= self.CRITICAL:
                cross = snap.alternator_v < 12.0
                sc = severity_score(1.0, 20, cross, self.hist.z_score(v))
                alerts.append(Alert("BATTERY", AlertLayer.CRITICAL, FaultCategory.DERIVED,
                    band(sc), sc, f"Crank voltage {v:.2f}V — no-start imminent. Replace NOW.",
                    v, self.CRITICAL)); return alerts
            if len(self.hist) >= 5:
                slope, _ = self.hist.linear_regression()
                if slope < -0.05:
                    cross = snap.alternator_v < 13.5
                    sc = severity_score(abs(slope)/0.2, len(self.hist), cross, self.hist.z_score(v))
                    alerts.append(Alert("BATTERY", AlertLayer.PREDICTIVE, FaultCategory.DERIVED,
                        band(sc), sc,
                        f"Crank V degrading {abs(slope):.3f}V/session. Health {self._health(v):.0f}%. "
                        f"Alternator output: {snap.alternator_v:.1f}V.",
                        v, self.WARN))
            z = self.hist.z_score(v)
            if v < self.HEALTHY and z < -1.8:
                sc = severity_score((self.HEALTHY-v)/2, self.gate.count, False, z)
                alerts.append(Alert("BATTERY", AlertLayer.ANOMALY, FaultCategory.DERIVED,
                    band(sc), sc,
                    f"Abnormal crank sag (Z={z:.2f}, {v:.2f}V). Ambient-corrected warn={warn_thr:.2f}V.",
                    v, self.HEALTHY))
        return alerts


class AlignmentDetector:
    STRAIGHT_TOL = 3.0; STRAIGHT_T = 10.0; DRIFT_THR = 0.5
    WARN_DEG = 4.0; CRIT_DEG = 6.0

    def __init__(self):
        self.kf_d = KalmanFilter1D(0.005, 0.2); self.kf_a = KalmanFilter1D(0.01, 1.0)
        self.log = MovingWindow(50); self.offset = 0.0
        self.straight_since: Optional[float] = None; self.gate = PersistenceGate(8)

    def analyze(self, snap: VehicleSnapshot) -> list[Alert]:
        alerts = []
        drift = self.kf_d.update(abs(snap.lateral_drift_m))
        angle = self.kf_a.update(abs(snap.steering_angle))
        now   = snap.timestamp

        if angle <= self.STRAIGHT_TOL:
            self.straight_since = self.straight_since or now
        else:
            self.straight_since = None

        straight_s = (now - self.straight_since) if self.straight_since else 0.0
        if straight_s >= self.STRAIGHT_T and drift > self.DRIFT_THR:
            self.offset += drift * 0.5
            self.log.push(self.offset)
            off = self.offset

            # Cross-sensor: check if individual wheel speeds differ
            wdiff = abs(snap.wheel_speed_fl - snap.wheel_speed_fr)
            cross = wdiff > 2.0

            if self.gate.update(True):
                if off >= self.CRIT_DEG:
                    sc = severity_score(1.0, self.gate.count, cross, self.log.z_score(off))
                    alerts.append(Alert("ALIGNMENT", AlertLayer.CRITICAL, FaultCategory.DERIVED,
                        band(sc), sc,
                        f"Offset {off:.1f}° — handling degraded. Fuel eff ↓4%. Immediate service.",
                        off, self.CRIT_DEG))
                elif off >= 1.0 and len(self.log) >= 5:
                    slope, _ = self.log.linear_regression()
                    eta = (self.WARN_DEG - off) / slope if slope > 0 else None
                    sc = severity_score(off/self.CRIT_DEG, self.gate.count, cross, self.log.z_score(off))
                    alerts.append(Alert("ALIGNMENT", AlertLayer.PREDICTIVE, FaultCategory.DERIVED,
                        band(sc), sc,
                        f"Drift {slope:.3f}°/event. Current {off:.2f}°. Wheel-speed diff {wdiff:.1f} km/h.",
                        off, self.WARN_DEG, eta))
                z = self.log.z_score(off)
                if z > 2.2:
                    sc = severity_score(z/4, self.gate.count, cross, z)
                    alerts.append(Alert("ALIGNMENT", AlertLayer.ANOMALY, FaultCategory.DERIVED,
                        band(sc), sc,
                        f"Sudden drift spike Z={z:.2f} — pothole/suspension event?",
                        off, self.WARN_DEG))
        else:
            self.gate.update(False)
        return alerts


class VibrationDetector:
    CRIT_RMS = 3.0; WARN_RMS = 1.5; Z_THR = 2.5; SLOPE_THR = 0.01

    def __init__(self):
        self.kf = KalmanFilter1D(0.005, 0.15)
        self.baseline = MovingWindow(50); self.trend = MovingWindow(20)
        self.gate = PersistenceGate(3); self.spikes = 0

    def analyze(self, snap: VehicleSnapshot) -> list[Alert]:
        # Rough road filter: ignore <200ms spikes (handled by persistence gate)
        alerts = []
        rms = self.kf.update(snap.vibration_rms)
        self.baseline.push(rms); self.trend.push(rms)
        z = self.baseline.z_score(rms)

        if rms >= self.CRIT_RMS:
            sc = severity_score(1.0, 20, z > 3.0, z)
            alerts.append(Alert("VIBRATION", AlertLayer.CRITICAL, FaultCategory.DIRECT,
                band(sc), sc,
                f"RMS {rms:.2f}g — imminent bearing failure. Reduce speed NOW.",
                rms, self.CRIT_RMS)); return alerts

        if len(self.trend) >= 10:
            slope, _ = self.trend.linear_regression()
            if self.gate.update(slope >= self.SLOPE_THR):
                eta = (self.WARN_RMS - rms) / slope if slope > 0 else None
                sc = severity_score(slope/0.05, self.gate.count, z > 1.5, z)
                alerts.append(Alert("VIBRATION", AlertLayer.PREDICTIVE, FaultCategory.DIRECT,
                    band(sc), sc,
                    f"Vibration baseline ↑ {slope:.4f}g/frame. Early bearing wear probable.",
                    rms, self.WARN_RMS, eta))
            else:
                self.gate.update(False)

        if z > self.Z_THR:
            self.spikes += 1
            if self.spikes >= 3:
                sc = severity_score(z/4, self.spikes, False, z)
                alerts.append(Alert("VIBRATION", AlertLayer.ANOMALY, FaultCategory.DIRECT,
                    band(sc), sc,
                    f"{self.spikes} spikes (Z={z:.2f}, {rms:.2f}g). Schedule axle inspection ≤500 miles.",
                    rms, self.baseline.mean() + self.Z_THR * self.baseline.std()))
        else:
            self.spikes = max(0, self.spikes - 1)
        return alerts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ══ HEAVY VEHICLE DETECTORS (J1939) ══
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AirBrakeDetector:
    """
    DIRECT FAULT — J1939 air brake pressure monitoring.
    Alert if primary OR secondary drops below 60 PSI, or fails to build at idle.
    """
    MIN_PSI = 60.0; BUILD_RATE_PSI_S = 2.0   # expected build rate while engine running

    def __init__(self):
        self.kf_p = KalmanFilter1D(0.01, 1.0); self.kf_s = KalmanFilter1D(0.01, 1.0)
        self.gate_p = PersistenceGate(3); self.gate_s = PersistenceGate(3)
        self.idle_window_p = MovingWindow(20); self.idle_window_s = MovingWindow(20)

    def analyze(self, snap: VehicleSnapshot) -> list[Alert]:
        alerts = []
        p = self.kf_p.update(snap.brake_air_pressure_primary)
        s = self.kf_s.update(snap.brake_air_pressure_secondary)

        for label, val, win, gate in [("PRIMARY",   p, self.idle_window_p, self.gate_p),
                                       ("SECONDARY", s, self.idle_window_s, self.gate_s)]:
            win.push(val)
            slope, _ = win.linear_regression()
            failing_to_build = snap.rpm > 600 and slope < -0.2 and val < 90

            if gate.update(val < self.MIN_PSI or failing_to_build):
                cross = (p < self.MIN_PSI and s < self.MIN_PSI)
                sc = severity_score((self.MIN_PSI - min(val, self.MIN_PSI)) / self.MIN_PSI,
                                    gate.count, cross, win.z_score(val))
                layer = AlertLayer.CRITICAL if val < self.MIN_PSI else AlertLayer.ANOMALY
                alerts.append(Alert(
                    f"AIR BRAKE [{label}]", layer, FaultCategory.DIRECT,
                    band(sc), sc,
                    f"Brake air pressure {val:.1f} PSI {'(BELOW SAFE LIMIT)' if val < self.MIN_PSI else '(failing to build)'}. "
                    f"Dual-circuit failure: {'YES' if cross else 'no'}. "
                    f"DO NOT operate vehicle until pressure is restored.",
                    val, self.MIN_PSI))
        return alerts


class DPFRegenDetector:
    """
    DERIVED FAULT — DPF clogging detection.
    High backpressure + low exhaust temp = filter blocked (regen not happening).
    """
    HIGH_BP_KPA = 10.0    # kPa  — elevated backpressure
    LOW_TEMP_C  = 300.0   # °C   — regen needs >550°C; below 300 = cold/no-regen

    def __init__(self):
        self.kf_bp = KalmanFilter1D(0.01, 0.5); self.kf_et = KalmanFilter1D(0.02, 5.0)
        self.gate  = PersistenceGate(2)

    def analyze(self, snap: VehicleSnapshot) -> list[Alert]:
        alerts = []
        bp = self.kf_bp.update(snap.dpf_backpressure_kpa)
        et = self.kf_et.update(snap.exhaust_temp_c)

        # Core logic: pressure HIGH + temp LOW = clogging (regen failing)
        clogging = bp > self.HIGH_BP_KPA and et < self.LOW_TEMP_C
        # Cross-sensor: SCR system also failing amplifies severity
        cross = not snap.scr_status_ok

        if self.gate.update(clogging):
            sc = severity_score((bp / self.HIGH_BP_KPA - 1) / 2,
                                 self.gate.count, cross, 0.0)
            alerts.append(Alert(
                "DPF REGEN", AlertLayer.ANOMALY, FaultCategory.DERIVED,
                band(sc), sc,
                f"DPF backpressure {bp:.1f} kPa (>{self.HIGH_BP_KPA}) but exhaust only {et:.0f}°C. "
                f"Filter clogging — forced regen required. SCR also fault: {'YES' if cross else 'no'}.",
                bp, self.HIGH_BP_KPA))
        return alerts


class AxleOverloadDetector:
    """
    DIRECT FAULT — Air suspension pressure → estimated axle load.
    Alert if estimated gross weight exceeds configured GVWR.
    Pressure-to-weight: 1 PSI ≈ 80 kg (approximate; calibrate per vehicle).
    """
    GVWR_KG       = 14000    # Gross Vehicle Weight Rating (configure per truck)
    PSI_TO_KG     = 80.0     # calibration factor
    NUM_AXLES     = 2         # drive + steer axle

    def __init__(self):
        self.kf   = KalmanFilter1D(0.01, 2.0)
        self.gate = PersistenceGate(3)

    def _est_weight(self, psi: float) -> float:
        return psi * self.PSI_TO_KG * self.NUM_AXLES

    def analyze(self, snap: VehicleSnapshot) -> list[Alert]:
        alerts = []
        psi    = self.kf.update(snap.air_susp_pressure_psi)
        weight = self._est_weight(psi)
        over   = weight - self.GVWR_KG

        if self.gate.update(over > 0):
            sc = severity_score(min(1.0, over / 2000), self.gate.count, True, 0.0)
            alerts.append(Alert(
                "AXLE OVERLOAD", AlertLayer.CRITICAL, FaultCategory.DIRECT,
                band(sc), sc,
                f"Estimated load {weight:.0f} kg exceeds GVWR {self.GVWR_KG} kg "
                f"(+{over:.0f} kg overage). Axle and brake stress risk.",
                weight, self.GVWR_KG))
        return alerts


class CoolingInefficDetector:
    """
    DERIVED FAULT — Radiator blockage detection.
    Fan at 100% but temp still rising under LOW load = radiator blocked.
    Suppressed on steep inclines (mountain driving expected to run hot).
    """
    FAN_FULL_PCT    = 95.0   # % — "fan is maxed out"
    TEMP_RISE_SLOPE = 0.2    # °C/frame — still rising despite full fan
    LOW_LOAD_THR    = 50.0   # % — should be easy to cool below this load

    def __init__(self):
        self.kf_t = KalmanFilter1D(0.02, 1.5)
        self.temp_win = MovingWindow(15)
        self.gate     = PersistenceGate(6)

    def analyze(self, snap: VehicleSnapshot) -> list[Alert]:
        if not ContextFilter.not_mountain_driving(snap): return []
        alerts = []
        t = self.kf_t.update(snap.coolant_temp); self.temp_win.push(t)

        if snap.fan_speed_pct >= self.FAN_FULL_PCT and snap.engine_load <= self.LOW_LOAD_THR:
            slope, _ = self.temp_win.linear_regression()
            if self.gate.update(slope >= self.TEMP_RISE_SLOPE):
                cross = snap.oil_temp > 95  # oil also heating up confirms blockage
                sc = severity_score(slope/0.5, self.gate.count, cross, self.temp_win.z_score(t))
                alerts.append(Alert(
                    "COOLING EFFIC", AlertLayer.ANOMALY, FaultCategory.DERIVED,
                    band(sc), sc,
                    f"Fan at {snap.fan_speed_pct:.0f}% but coolant still rising {slope:.2f}°C/frame "
                    f"at only {snap.engine_load:.0f}% load. Radiator blockage likely. "
                    f"Oil-temp corroboration: {'YES' if cross else 'no'}.",
                    t, 95.0))
        else:
            self.gate.update(False)
        return alerts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ══ CAR-SPECIFIC DETECTORS ══
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MisfireDetector:
    """
    DIRECT FAULT — Crank interval irregularity + O2 sensor correlation.
    A healthy engine fires every ~N ms (depends on RPM and cylinders).
    Irregular intervals = misfire. Confirmed by rich/lean O2 spike.
    """
    INTERVAL_STD_THR   = 1.2    # ms std-dev on crank intervals
    O2_LEAN_V          = 0.2    # V — lean spike after misfire (raw fuel spike)
    O2_RICH_V          = 0.8    # V — rich spike
    MIN_COOLANT_FOR_DIAG = 79   # °C — cold start immunity

    def __init__(self):
        self.crank_win = MovingWindow(20)
        self.o2_win    = MovingWindow(20)
        self.gate      = PersistenceGate(1)

    def analyze(self, snap: VehicleSnapshot) -> list[Alert]:
        if snap.coolant_temp < self.MIN_COOLANT_FOR_DIAG: return []
        if snap.rpm < 500: return []
        alerts = []

        self.crank_win.push(snap.crank_interval_ms)
        self.o2_win.push(snap.o2_sensor_voltage)
        std = self.crank_win.std()

        o2_abnormal = (snap.o2_sensor_voltage < self.O2_LEAN_V or
                       snap.o2_sensor_voltage > self.O2_RICH_V)
        cross = o2_abnormal  # crankshaft + O2 both flagged = confirmed misfire

        if self.gate.update(std > self.INTERVAL_STD_THR):
            z = self.crank_win.z_score(snap.crank_interval_ms)
            sc = severity_score(std / 10.0, self.gate.count, cross,
                                self.o2_win.z_score(snap.o2_sensor_voltage))
            layer = AlertLayer.CRITICAL if sc >= 71 else AlertLayer.ANOMALY
            alerts.append(Alert(
                "MISFIRE", layer, FaultCategory.DIRECT,
                band(sc), sc,
                f"Crank interval std {std:.2f}ms (>{self.INTERVAL_STD_THR}ms). "
                f"O2={snap.o2_sensor_voltage:.2f}V {'(ABNORMAL ✗)' if o2_abnormal else '(OK)'}. "
                f"Fuel trim {snap.fuel_trim:+.1f}%.",
                std, self.INTERVAL_STD_THR))
        return alerts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ══ TWO-WHEELER DETECTORS ══
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TipOverDetector:
    """
    DIRECT FAULT — Tip-over / crash detection.
    High-G impact followed by sustained 90° lean = fallen bike.
    """
    LEAN_CRIT_DEG  = 70.0    # degrees from vertical — fallen
    IMPACT_G       = 3.0     # g — sudden impact threshold
    SUSTAIN_FRAMES = 3        # frames of sustained lean to confirm

    def __init__(self):
        self.kf_lean = KalmanFilter1D(0.02, 2.0)
        self.kf_acc  = KalmanFilter1D(0.01, 0.3)
        self.gate    = PersistenceGate(2)
        self.impact_detected = False

    def analyze(self, snap: VehicleSnapshot) -> list[Alert]:
        alerts = []
        lean = self.kf_lean.update(abs(snap.lean_angle_deg))
        acc  = self.kf_acc.update(math.sqrt(
            snap.lateral_accel**2 + snap.longitudinal_accel**2 + snap.vertical_accel**2))

        if acc > self.IMPACT_G:
            self.impact_detected = True

        if self.gate.update(lean >= self.LEAN_CRIT_DEG):
            sc = severity_score(lean / 90.0, self.gate.count, self.impact_detected, 0.0)
            layer = AlertLayer.CRITICAL if self.impact_detected else AlertLayer.ANOMALY
            alerts.append(Alert(
                "TIP-OVER", layer, FaultCategory.DIRECT,
                band(sc), sc,
                f"Lean angle {lean:.1f}° sustained for {self.gate.count} frames. "
                f"Impact detected: {'YES — possible crash!' if self.impact_detected else 'no (slow tip)'}. "
                f"Peak G: {acc:.2f}.",
                lean, self.LEAN_CRIT_DEG))
        else:
            if lean < 20: self.impact_detected = False  # reset after recovery
        return alerts


class BeltChainSlipDetector:
    """
    DERIVED FAULT — Drive belt/chain slip.
    RPM rises but rear wheel speed doesn't follow the expected gear ratio.
    """
    GEAR_RATIOS   = {1: 12.0, 2: 7.5, 3: 5.5, 4: 4.2, 5: 3.3}   # RPM / (wheel km/h)
    SLIP_RATIO    = 0.15   # 15% deviation from expected = slip

    def __init__(self):
        self.kf_rpm = KalmanFilter1D(0.05, 50.0)
        self.kf_ws  = KalmanFilter1D(0.02, 1.0)
        self.gate   = PersistenceGate(5)
        self.rpm_win = MovingWindow(10); self.ws_win = MovingWindow(10)

    def _infer_gear(self, rpm: float, wheel_kmh: float) -> Optional[int]:
        if wheel_kmh < 5: return None
        ratio = rpm / wheel_kmh
        return min(self.GEAR_RATIOS, key=lambda g: abs(self.GEAR_RATIOS[g] - ratio))

    def analyze(self, snap: VehicleSnapshot) -> list[Alert]:
        alerts = []
        rpm = self.kf_rpm.update(snap.rpm)
        ws  = self.kf_ws.update(snap.wheel_speed_rear)
        self.rpm_win.push(rpm); self.ws_win.push(ws)

        if ws < 5 or rpm < 500: return []

        gear  = self._infer_gear(rpm, ws)
        if gear is None: return []
        exp_ratio = self.GEAR_RATIOS[gear]
        actual    = rpm / ws if ws > 0 else 0
        deviation = abs(actual - exp_ratio) / exp_ratio

        # RPM climbing fast but wheel speed flat = slip
        rpm_slope, _ = self.rpm_win.linear_regression()
        ws_slope,  _ = self.ws_win.linear_regression()
        rpm_up_ws_flat = rpm_slope > 50 and ws_slope < 1.0

        if self.gate.update(deviation > self.SLIP_RATIO or rpm_up_ws_flat):
            sc = severity_score(deviation / 0.5, self.gate.count, rpm_up_ws_flat, 0.0)
            alerts.append(Alert(
                "BELT/CHAIN SLIP", AlertLayer.ANOMALY, FaultCategory.DERIVED,
                band(sc), sc,
                f"Gear {gear}: expected ratio {exp_ratio:.1f}, actual {actual:.1f} "
                f"({deviation*100:.1f}% deviation). RPM rising while wheel speed flat: "
                f"{'YES' if rpm_up_ws_flat else 'no'}.",
                deviation, self.SLIP_RATIO))
        return alerts


class BrakeWearDetector:
    """
    DERIVED FAULT — Brake pad wear via deceleration efficiency trend.
    If lever is pressed longer / harder to achieve the same speed reduction,
    friction material is worn.
    Baseline = decel_rate / brake_duration during first N braking events.
    """
    MIN_EVENTS    = 5
    WEAR_THR      = 0.25   # 25% drop in efficiency = alert

    def __init__(self):
        self.efficiency_log: list[float] = []
        self.baseline_eff: Optional[float] = None
        self.gate = PersistenceGate(3)

    def _efficiency(self, decel: float, duration: float) -> Optional[float]:
        if duration < 0.1: return None
        return decel / duration  # m/s² per second of lever press

    def analyze(self, snap: VehicleSnapshot) -> list[Alert]:
        alerts = []
        if not snap.brake_lever_pressed or snap.decel_rate_ms2 < 0.1: return []

        eff = self._efficiency(snap.decel_rate_ms2, snap.brake_duration_s)
        if eff is None: return []

        self.efficiency_log.append(eff)

        if len(self.efficiency_log) < self.MIN_EVENTS: return []

        if self.baseline_eff is None:
            self.baseline_eff = sum(self.efficiency_log[:self.MIN_EVENTS]) / self.MIN_EVENTS

        drop = (self.baseline_eff - eff) / self.baseline_eff

        if self.gate.update(drop >= self.WEAR_THR):
            sc = severity_score(drop, self.gate.count, False, 0.0)
            alerts.append(Alert(
                "BRAKE WEAR", AlertLayer.PREDICTIVE, FaultCategory.DERIVED,
                band(sc), sc,
                f"Braking efficiency dropped {drop*100:.1f}% from baseline "
                f"({self.baseline_eff:.2f} → {eff:.2f} m/s² per sec). "
                f"Pad wear likely. Inspection recommended.",
                eff, self.baseline_eff * (1 - self.WEAR_THR)))
        return alerts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HEALTH SCORE ENGINE  (per vehicle class)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HEALTH_WEIGHTS: dict[VehicleClass, dict[str, float]] = {
    VehicleClass.HEAVY: {
        "ENGINE TEMP":   0.15, "BATTERY":       0.10,
        "ALIGNMENT":     0.05, "VIBRATION":     0.10,
        "AIR BRAKE [PRIMARY]":   0.20, "AIR BRAKE [SECONDARY]": 0.15,
        "DPF REGEN":     0.10, "AXLE OVERLOAD": 0.10, "COOLING EFFIC": 0.05,
    },
    VehicleClass.CAR: {
        "ENGINE TEMP":  0.30, "BATTERY":     0.25,
        "ALIGNMENT":    0.20, "VIBRATION":   0.15,
        "MISFIRE":      0.10,
    },
    VehicleClass.TWO_WHEELER: {
        "ENGINE TEMP":  0.20, "BATTERY":     0.10,
        "ALIGNMENT":    0.10, "VIBRATION":   0.15,
        "TIP-OVER":     0.25, "BELT/CHAIN SLIP": 0.10, "BRAKE WEAR": 0.10,
    },
}

LAYER_PENALTY = {AlertLayer.PREDICTIVE: 3, AlertLayer.ANOMALY: 12, AlertLayer.CRITICAL: 35}


class HealthScoreEngine:
    def __init__(self, vc: VehicleClass):
        self.weights = HEALTH_WEIGHTS[vc]
        self.scores  = {k: 100.0 for k in self.weights}

    def update(self, alerts: list[Alert]):
        active = {a.module for a in alerts}
        for a in alerts:
            if a.module in self.scores:
                self.scores[a.module] = max(0.0, self.scores[a.module] - LAYER_PENALTY[a.layer])
        for m in self.scores:
            if m not in active:
                self.scores[m] = min(100.0, self.scores[m] + 0.6)

    def score(self) -> float:
        s = sum(self.scores.get(m, 100.0) * w for m, w in self.weights.items())
        return max(0.0, min(100.0, s))

    def band_str(self, s: float) -> str:
        if s >= 85: return "✅ GOOD"
        if s >= 65: return "🟡 FAIR"
        if s >= 40: return "🟠 DEGRADED"
        return "🔴 CRITICAL"

    def top_modules(self, n: int = 3) -> str:
        worst = sorted(self.scores.items(), key=lambda x: x[1])[:n]
        return "  ".join(f"{m[:6]}:{v:.0f}" for m, v in worst)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MASTER PREDICTIVE SAFETY SYSTEM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PredictiveSafetySystem:
    """
    Instantiate with a VehicleClass. Call .process(snap) each sensor frame.
    Returns (health_score: float, alerts: list[Alert]).
    """
    def __init__(self, vehicle_class: VehicleClass = VehicleClass.CAR):
        self.vc    = vehicle_class
        self.frame = 0

        # ── Shared detectors (all classes) ──────────
        self.eng  = EngineTempDetector()
        self.bat  = BatteryDetector()
        self.aln  = AlignmentDetector()
        self.vib  = VibrationDetector()

        # ── Heavy ────────────────────────────────────
        self.air_brake    = AirBrakeDetector()      if self.vc == VehicleClass.HEAVY else None
        self.dpf          = DPFRegenDetector()      if self.vc == VehicleClass.HEAVY else None
        self.axle         = AxleOverloadDetector()  if self.vc == VehicleClass.HEAVY else None
        self.cool_ineff   = CoolingInefficDetector()if self.vc == VehicleClass.HEAVY else None

        # ── Car ──────────────────────────────────────
        self.misfire      = MisfireDetector()       if self.vc == VehicleClass.CAR   else None

        # ── Two-wheeler ───────────────────────────────
        self.tipover      = TipOverDetector()       if self.vc == VehicleClass.TWO_WHEELER else None
        self.belt         = BeltChainSlipDetector() if self.vc == VehicleClass.TWO_WHEELER else None
        self.brake_wear   = BrakeWearDetector()     if self.vc == VehicleClass.TWO_WHEELER else None

        self.scorer = HealthScoreEngine(vehicle_class)

    def process(self, snap: VehicleSnapshot) -> tuple[float, list[Alert]]:
        self.frame += 1
        alerts: list[Alert] = []

        alerts += self.eng.analyze(snap)
        alerts += self.bat.analyze(snap)
        alerts += self.aln.analyze(snap)
        alerts += self.vib.analyze(snap)

        for det in [self.air_brake, self.dpf, self.axle, self.cool_ineff,
                    self.misfire, self.tipover, self.belt, self.brake_wear]:
            if det: alerts += det.analyze(snap)

        # Sort: CRITICAL → ANOMALY → PREDICTIVE; within tier, by score desc
        priority = {AlertLayer.CRITICAL: 0, AlertLayer.ANOMALY: 1, AlertLayer.PREDICTIVE: 2}
        alerts.sort(key=lambda a: (priority[a.layer], -a.score))

        self.scorer.update(alerts)
        return self.scorer.score(), alerts

    def report(self, snap: VehicleSnapshot, score: float, alerts: list[Alert]):
        lbl    = self.scorer.band_str(score)
        filled = int(score / 5)
        bar    = "█" * filled + "░" * (20 - filled)

        print(f"\n{'═'*72}")
        print(f"  {self.vc.value}  │  FRAME #{self.frame:04d}  │  {lbl}  [{bar}] {score:.1f}/100")
        print(f"  Worst modules ▸ {self.scorer.top_modules()}")
        print(f"{'─'*72}")
        print(
            f"  🌡  {snap.coolant_temp:.1f}°C  Load:{snap.engine_load:.0f}%  Amb:{snap.ambient_temp:.0f}°C  "
            f"Oil:{snap.oil_temp:.0f}°C  Fan:{snap.fan_speed_pct:.0f}%\n"
            f"  🔋  {snap.battery_voltage:.2f}V  Alt:{snap.alternator_v:.1f}V  "
            f"Crank:{'YES' if snap.is_cranking else 'no'}\n"
            f"  🚗  Drift:{snap.lateral_drift_m:.2f}m  Steer:{snap.steering_angle:.1f}°  "
            f"Speed:{snap.speed_kmh:.0f}km/h  RPM:{snap.rpm:.0f}\n"
            f"  📳  Vib:{snap.vibration_rms:.3f}g  "
            f"Lean:{snap.lean_angle_deg:.1f}°  Incline:{snap.incline_deg:.1f}°"
        )
        if self.vc == VehicleClass.HEAVY:
            print(
                f"  🛑  BrakeP:{snap.brake_air_pressure_primary:.0f}/{snap.brake_air_pressure_secondary:.0f}PSI  "
                f"DPF:{snap.dpf_backpressure_kpa:.1f}kPa  Exh:{snap.exhaust_temp_c:.0f}°C  "
                f"SuspPSI:{snap.air_susp_pressure_psi:.0f}"
            )
        if alerts:
            print(f"\n  ┌── {len(alerts)} ALERT(S) ─────────────────────────────────────────────")
            for a in alerts:
                print(str(a))
            print(f"  └{'─'*66}")
        else:
            print("\n  ✔  All systems nominal — no anomalies detected.")
        print(f"{'═'*72}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SCENARIO FACTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _s(**kw) -> VehicleSnapshot:
    base = dict(rpm=2200, speed_kmh=60, engine_load=30, coolant_temp=85, oil_temp=83,
                battery_voltage=12.6, alternator_v=14.2, fan_speed_pct=50,
                vibration_rms=0.08, lateral_drift_m=0, steering_angle=0,
                wheel_speed_fl=60, wheel_speed_fr=60, ambient_temp=32,
                brake_air_pressure_primary=120, brake_air_pressure_secondary=120,
                dpf_backpressure_kpa=5, exhaust_temp_c=420, scr_status_ok=True,
                air_susp_pressure_psi=100, incline_deg=0, lean_angle_deg=0,
                lateral_accel=0, longitudinal_accel=0, vertical_accel=1.0,
                wheel_speed_rear=60, crank_interval_ms=50, o2_sensor_voltage=0.45,
                fuel_trim=0, brake_lever_pressed=False, brake_duration_s=0,
                decel_rate_ms2=0, is_cranking=False, timestamp=time.time())
    base.update(kw)
    return VehicleSnapshot(**base)


# ── HEAVY scenarios ──────────────────────────────────────────────────────────

def scen_air_brake_leak(n=12):
    base = time.time()
    return [_s(brake_air_pressure_primary=120 - i*7 + random.gauss(0, 1),
               brake_air_pressure_secondary=118 - i*3 + random.gauss(0, 1),
               rpm=700, timestamp=base+i*60) for i in range(n)]

def scen_dpf_clogging(n=12):
    base = time.time()
    return [_s(dpf_backpressure_kpa=4 + i*1.2 + random.gauss(0, 0.2),
               exhaust_temp_c=230 - i*3 + random.gauss(0, 5),
               scr_status_ok=(i < 8), timestamp=base+i*60) for i in range(n)]

def scen_axle_overload(n=8):
    base = time.time()
    return [_s(air_susp_pressure_psi=95 + i*8 + random.gauss(0, 2),
               timestamp=base+i*5) for i in range(n)]

def scen_radiator_blockage(n=14):
    base = time.time()
    return [_s(coolant_temp=88 + i*0.9 + random.gauss(0, 0.3),
               oil_temp=86 + i*0.8 + random.gauss(0, 0.3),
               fan_speed_pct=96, engine_load=25, incline_deg=2,
               timestamp=base+i*60) for i in range(n)]

# ── CAR scenarios ────────────────────────────────────────────────────────────

def scen_engine_overheat(n=25):
    base = time.time()
    return [_s(coolant_temp=83 + i*0.9 + random.gauss(0, 0.4),
               oil_temp=81 + i*0.85 + random.gauss(0, 0.3),
               engine_load=28, ambient_temp=36, timestamp=base+i*60) for i in range(n)]

def scen_misfire(n=14):
    base = time.time()
    snaps = []
    for i in range(n):
        fire = random.gauss(0, 0.5 + i*0.15)   # increasing irregularity
        snaps.append(_s(crank_interval_ms=50 + fire,
                        o2_sensor_voltage=0.45 + (0.4 if i > 7 else 0)*random.choice([-1,1]),
                        fuel_trim=i*0.8, coolant_temp=80, timestamp=base+i*30))
    return snaps

def scen_battery_degradation(n=15):
    base = time.time()
    return [_s(battery_voltage=9.9 - i*0.07 + random.gauss(0, 0.03),
               alternator_v=14.2 - i*0.05,
               is_cranking=(i % 3 == 0), timestamp=base+i*86400) for i in range(n)]

# ── TWO-WHEELER scenarios ────────────────────────────────────────────────────

def scen_tip_over(n=8):
    base = time.time()
    return [_s(lean_angle_deg=i*11 + random.gauss(0, 2),
               lateral_accel=(3.5 if i == 3 else 0.1),
               vertical_accel=(0.3 if i > 3 else 1.0),
               timestamp=base+i*2) for i in range(n)]

def scen_belt_slip(n=14):
    base = time.time()
    return [_s(rpm=3000 + i*100 + random.gauss(0, 50),
               wheel_speed_rear=max(5, 60 - i*1.5 + random.gauss(0, 1)),
               speed_kmh=max(5, 60 - i*1.5), timestamp=base+i*30) for i in range(n)]

def scen_brake_wear(n=15):
    base = time.time()
    snaps = []
    for i in range(n):
        decel = max(0.5, 6.0 - i*0.3 + random.gauss(0, 0.2))  # efficiency falling
        dur   = 1.0 + i*0.1                                      # pressing longer
        snaps.append(_s(brake_lever_pressed=True, decel_rate_ms2=decel,
                        brake_duration_s=dur, speed_kmh=50, timestamp=base+i*300))
    return snaps

def scen_compound_car(n=18):
    base = time.time()
    return [_s(coolant_temp=87+i*0.75+random.gauss(0,0.4),
               oil_temp=85+i*0.7+random.gauss(0,0.3),
               engine_load=35, ambient_temp=38,
               battery_voltage=9.8-i*0.06+random.gauss(0,0.03),
               alternator_v=14.2-i*0.04,
               is_cranking=(i%4==0),
               lateral_drift_m=0.45+i*0.06+random.gauss(0,0.04),
               steering_angle=random.gauss(0,1.8),
               vibration_rms=0.07+i*0.03+(random.uniform(0.4,0.9) if i>10 else 0),
               crank_interval_ms=50+random.gauss(0,0.3+i*0.18),
               o2_sensor_voltage=0.45+(0.35 if i>12 else 0)*random.choice([-1,1]),
               timestamp=base+i*86400) for i in range(n)]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RUNNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(title: str, vc: VehicleClass, snaps: list[VehicleSnapshot]):
    print(f"\n\n{'#'*72}")
    print(f"  {vc.value}  │  {title}")
    print(f"{'#'*72}")
    sys = PredictiveSafetySystem(vc)
    for snap in snaps:
        score, alerts = sys.process(snap)
        sys.report(snap, score, alerts)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║         UNIFIED PREDICTIVE VEHICLE SAFETY SYSTEM  v3.0                     ║
║         HEAVY · CAR · TWO-WHEELER  |  Direct · Derived · Smart Validation  ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # ── Heavy ────────────────────────────────────────────────────────────────
    run("AIR BRAKE LEAK",        VehicleClass.HEAVY,       scen_air_brake_leak())
    run("DPF / REGEN FAILURE",   VehicleClass.HEAVY,       scen_dpf_clogging())
    run("AXLE OVERLOAD",         VehicleClass.HEAVY,       scen_axle_overload())
    run("RADIATOR BLOCKAGE",     VehicleClass.HEAVY,       scen_radiator_blockage())

    # ── Car ──────────────────────────────────────────────────────────────────
    run("ENGINE OVERHEAT TREND", VehicleClass.CAR,         scen_engine_overheat())
    run("CYLINDER MISFIRE",      VehicleClass.CAR,         scen_misfire())
    run("BATTERY DEGRADATION",   VehicleClass.CAR,         scen_battery_degradation())
    run("COMPOUND FAULT",        VehicleClass.CAR,         scen_compound_car())

    # ── Two-Wheeler ───────────────────────────────────────────────────────────
    run("TIP-OVER / CRASH",      VehicleClass.TWO_WHEELER, scen_tip_over())
    run("BELT / CHAIN SLIP",     VehicleClass.TWO_WHEELER, scen_belt_slip())
    run("BRAKE WEAR",            VehicleClass.TWO_WHEELER, scen_brake_wear())

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  REAL-DATA INTEGRATION                                                      ║
║                                                                             ║
║  system = PredictiveSafetySystem(VehicleClass.HEAVY)   # or CAR / TWO_WHEEL║
║                                                                             ║
║  while True:                                                                ║
║      snap = VehicleSnapshot(                                                ║
║          coolant_temp   = obd.coolant_temp,                                 ║
║          engine_load    = obd.engine_load,                                  ║
║          oil_temp       = obd.oil_temp,                                     ║
║          battery_voltage= elm327.voltage,                                   ║
║          alternator_v   = elm327.alternator,                                ║
║          vibration_rms  = imu.rms_g,                                        ║
║          lean_angle_deg = imu.lean,          # two-wheeler only             ║
║          dpf_backpressure_kpa = j1939.dpf_bp,# heavy only                  ║
║          brake_air_pressure_primary = j1939.air_p,                          ║
║          ...                                                                ║
║      )                                                                      ║
║      score, alerts = system.process(snap)                                   ║
║      # Push to CAN bus / dashboard / mobile notification                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
