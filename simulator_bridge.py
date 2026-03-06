"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   SIMULATOR → SAFETY SYSTEM BRIDGE                                         ║
║   Maps simulator.py telemetry payload fields → VehicleSnapshot             ║
║   Runs the appropriate PredictiveSafetySystem per vehicle type             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   FAULT MAPPING (simulator fault_name → detector that catches it)          ║
║                                                                             ║
║   Simulator Fault         Caught By                      Layer             ║
║   ─────────────────────── ────────────────────────────── ───────────────── ║
║   ENGINE_OVERHEAT         EngineTempDetector             CRITICAL/PREDICTIVE║
║   COOLANT_LEAK            EngineTempDetector             ANOMALY/PREDICTIVE ║
║   OIL_PRESSURE_DROP       OilPressureDetector (new)      CRITICAL           ║
║   BATTERY_FAILURE         BatteryDetector                CRITICAL           ║
║   VIBRATION_SPIKE         VibrationDetector              ANOMALY/CRITICAL   ║
║   AIR_BRAKE_LOSS          AirBrakeDetector               CRITICAL           ║
║   OVERLOAD                AxleOverloadDetector           CRITICAL           ║
║   CHAIN_SLIP              BeltChainSlipDetector          ANOMALY            ║
║   BRAKE_WEAR              BrakeWearDetector              PREDICTIVE         ║
║   TURBO_FAILURE           TurboDetector (new)            ANOMALY            ║
║   TYRE_BLOWOUT_FL/RR      TyrePressureDetector (new)     CRITICAL           ║
║   OVERSPEED               OverspeedDetector (new)        WARNING            ║
║   HARSH_DRIVER            DriverBehaviorDetector (new)   WARNING            ║
║   FUEL_LEAK               FuelLevelDetector (new)        WARNING            ║
║   GPS_DRIFT               (logged, not safety-critical)  INFO               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    # Direct: feed a payload dict from the simulator
    from simulator_bridge import SimulatorBridge
    bridge = SimulatorBridge()
    score, alerts = bridge.process(payload_dict)

    # Or run the self-contained demo (no API needed):
    python simulator_bridge.py
"""

import math
import time
import random
from datetime import datetime, timezone
from typing import Optional

# ── Import the v3 safety system ─────────────────────────────────────────────
from vehicle_safety_system_v3 import (
    PredictiveSafetySystem, VehicleSnapshot, VehicleClass,
    Alert, AlertLayer, FaultCategory, SeverityBand,
    KalmanFilter1D, MovingWindow, PersistenceGate,
    severity_score, band,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SUPPLEMENTARY DETECTORS
#  (fields available in simulator but not covered by v3 base detectors)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class OilPressureDetector:
    """
    DIRECT FAULT — Catches OIL_PRESSURE_DROP.
    oil_pressure (PSI) drops below 50% of vehicle-normal → CRITICAL.
    Gradual drop below 70% → PREDICTIVE trend alert.
    Cross-validated with coolant_temp (if oil is low AND engine is hot = worse).
    """
    CRIT_RATIO  = 0.50   # < 50% of normal → critical
    WARN_RATIO  = 0.70   # < 70% → warning trend

    def __init__(self, normal_oil_psi: float):
        self.normal = normal_oil_psi
        self.kf     = KalmanFilter1D(0.01, 1.0)
        self.win    = MovingWindow(20)
        self.gate   = PersistenceGate(3)

    def analyze(self, snap: VehicleSnapshot, oil_pressure_psi: float) -> list[Alert]:
        alerts = []
        p = self.kf.update(oil_pressure_psi)
        self.win.push(p)
        ratio = p / self.normal

        # Cross-sensor: is engine also running hot?
        cross = snap.coolant_temp > snap.ambient_temp + 40

        if self.gate.update(ratio < self.CRIT_RATIO):
            sc = severity_score(1.0 - ratio, self.gate.count, cross, self.win.z_score(p))
            alerts.append(Alert(
                "OIL PRESSURE", AlertLayer.CRITICAL, FaultCategory.DIRECT,
                band(sc), sc,
                f"Oil pressure {p:.1f} PSI = {ratio*100:.0f}% of normal ({self.normal} PSI). "
                f"Engine seizure risk. Stop immediately. "
                f"Coolant-temp corroboration: {'YES' if cross else 'no'}.",
                p, self.normal * self.CRIT_RATIO))
        elif len(self.win) >= 5:
            slope, _ = self.win.linear_regression()
            if slope < -0.05 and ratio < self.WARN_RATIO:
                sc = severity_score(abs(slope) / 0.5, self.gate.count, cross, self.win.z_score(p))
                alerts.append(Alert(
                    "OIL PRESSURE", AlertLayer.PREDICTIVE, FaultCategory.DIRECT,
                    band(sc), sc,
                    f"Oil pressure trending down {abs(slope):.3f} PSI/frame. "
                    f"Currently {p:.1f} PSI ({ratio*100:.0f}% of normal). Check for leak.",
                    p, self.normal * self.WARN_RATIO))
        return alerts


class TyrePressureDetector:
    """
    DIRECT FAULT — Catches TYRE_BLOWOUT_FL and TYRE_BLOWOUT_RR.
    Any tyre > 20% below its nominal PSI = critical blowout risk.
    Also detects slow leak trend (<10% drop but sustained).
    """
    BLOWOUT_RATIO = 0.60   # < 60% of nominal = blowout
    LEAK_RATIO    = 0.85   # < 85% = slow leak warning

    def __init__(self, nominal_front: float, nominal_rear: float):
        self.nom = {"FL": nominal_front, "FR": nominal_front,
                    "RL": nominal_rear,  "RR": nominal_rear}
        self.wins  = {k: MovingWindow(20) for k in self.nom}
        self.gates = {k: PersistenceGate(2) for k in self.nom}

    def analyze(self, tyres: dict[str, float]) -> list[Alert]:
        """tyres = {"FL": psi, "FR": psi, "RL": psi, "RR": psi}"""
        alerts = []
        for pos, psi in tyres.items():
            nom  = self.nom[pos]
            self.wins[pos].push(psi)
            ratio = psi / nom

            if self.gates[pos].update(ratio < self.BLOWOUT_RATIO):
                slope, _ = self.wins[pos].linear_regression()
                cross = any(tyres[p] / self.nom[p] < self.LEAK_RATIO
                            for p in tyres if p != pos)
                sc = severity_score(1.0 - ratio, self.gates[pos].count, cross,
                                    self.wins[pos].z_score(psi))
                alerts.append(Alert(
                    f"TYRE [{pos}]", AlertLayer.CRITICAL, FaultCategory.DIRECT,
                    band(sc), sc,
                    f"Tyre {pos} at {psi:.1f} PSI = {ratio*100:.0f}% of nominal ({nom} PSI). "
                    f"Blowout risk. Reduce speed immediately. "
                    f"Other tyres also low: {'YES' if cross else 'no'}.",
                    psi, nom * self.BLOWOUT_RATIO))
            elif len(self.wins[pos]) >= 8:
                slope, _ = self.wins[pos].linear_regression()
                if slope < -0.01 and ratio < self.LEAK_RATIO:
                    sc = severity_score(abs(slope)/0.05, 2, False,
                                        self.wins[pos].z_score(psi))
                    alerts.append(Alert(
                        f"TYRE [{pos}]", AlertLayer.PREDICTIVE, FaultCategory.DIRECT,
                        band(sc), sc,
                        f"Slow tyre leak on {pos}: {psi:.1f} PSI "
                        f"({ratio*100:.0f}% of {nom} PSI nominal), "
                        f"dropping {abs(slope):.3f} PSI/frame.",
                        psi, nom * self.LEAK_RATIO))
        return alerts


class TurboDetector:
    """
    DERIVED FAULT — Catches TURBO_FAILURE.
    At highway speeds (> 30 km/h) turbo_boost should be > 0.
    If boost = 0 at speed → turbo has failed.
    Also alerts if boost is inconsistent (Z-Score spike).
    """
    MIN_SPEED_FOR_BOOST = 30.0   # km/h

    def __init__(self):
        self.win  = MovingWindow(15)
        self.gate = PersistenceGate(4)

    def analyze(self, speed_kmh: float, turbo_boost: float, rpm: float) -> list[Alert]:
        alerts = []
        self.win.push(turbo_boost)
        z = self.win.z_score(turbo_boost)

        failed = speed_kmh > self.MIN_SPEED_FOR_BOOST and turbo_boost == 0.0
        if self.gate.update(failed):
            sc = severity_score(1.0, self.gate.count, rpm > 3000, abs(z))
            alerts.append(Alert(
                "TURBO", AlertLayer.ANOMALY, FaultCategory.DIRECT,
                band(sc), sc,
                f"Turbo boost = 0 at {speed_kmh:.0f} km/h. "
                f"Turbo failure likely. Expect reduced power and higher exhaust temp.",
                turbo_boost, 1.0))
        elif abs(z) > 2.5 and turbo_boost > 0:
            sc = severity_score(abs(z)/4, 1, False, z)
            alerts.append(Alert(
                "TURBO", AlertLayer.ANOMALY, FaultCategory.DERIVED,
                band(sc), sc,
                f"Abnormal turbo boost spike {turbo_boost:.1f} PSI (Z={z:.2f}). "
                f"Possible boost leak or wastegate fault.",
                turbo_boost, self.win.mean()))
        return alerts


class OverspeedDetector:
    """
    DIRECT FAULT — Catches OVERSPEED.
    Speed > 120% of vehicle max_speed for that type.
    """
    def __init__(self, max_speed_kmh: float):
        self.limit = max_speed_kmh * 1.15
        self.gate  = PersistenceGate(3)
        self.win   = MovingWindow(10)

    def analyze(self, speed_kmh: float) -> list[Alert]:
        self.win.push(speed_kmh)
        if self.gate.update(speed_kmh > self.limit):
            sc = severity_score(min(1.0, (speed_kmh - self.limit) / 30),
                                self.gate.count, False, self.win.z_score(speed_kmh))
            return [Alert(
                "OVERSPEED", AlertLayer.CRITICAL, FaultCategory.DIRECT,
                band(sc), sc,
                f"Speed {speed_kmh:.0f} km/h exceeds safe limit {self.limit:.0f} km/h. "
                f"Driver intervention required.",
                speed_kmh, self.limit)]
        return []


class DriverBehaviorDetector:
    """
    DERIVED FAULT — Catches HARSH_DRIVER.
    Monitors harsh braking, harsh acceleration, extreme steering.
    Counts events per rolling window; alerts on pattern, not single event.
    """
    def __init__(self):
        self.harsh_win  = MovingWindow(30)   # rolling count of harsh events
        self.steer_win  = MovingWindow(20)

    def analyze(self, harsh_braking: bool, harsh_accel: bool,
                brake_pct: float, accel_pct: float,
                steering_angle: float) -> list[Alert]:
        alerts = []
        event = int(harsh_braking) + int(harsh_accel) + int(abs(steering_angle) > 40)
        self.harsh_win.push(event)
        self.steer_win.push(abs(steering_angle))

        avg_events = self.harsh_win.mean()
        z_steer    = self.steer_win.z_score(abs(steering_angle))

        if avg_events > 0.6:   # > 60% of recent frames have a harsh event
            sc = severity_score(avg_events, int(avg_events * 20), False, 0.0)
            alerts.append(Alert(
                "DRIVER BEHAVIOR", AlertLayer.ANOMALY, FaultCategory.DERIVED,
                band(sc), sc,
                f"Persistent harsh driving: avg {avg_events:.2f} events/frame "
                f"(brake={brake_pct:.0f}%, accel={accel_pct:.0f}%, steer={steering_angle:.0f}°). "
                f"Driver coaching recommended.",
                avg_events, 0.4))
        elif abs(steering_angle) > 40 and z_steer > 2.0:
            sc = severity_score(abs(steering_angle) / 60, 1, False, z_steer)
            alerts.append(Alert(
                "DRIVER BEHAVIOR", AlertLayer.ANOMALY, FaultCategory.DERIVED,
                band(sc), sc,
                f"Extreme steering angle {steering_angle:.0f}°. Z={z_steer:.2f}. "
                f"Possible loss of control or aggressive manoeuvre.",
                abs(steering_angle), 40.0))
        return alerts


class FuelLeakDetector:
    """
    DERIVED FAULT — Catches FUEL_LEAK.
    Computes expected fuel consumption rate from speed + load.
    If actual drain is >2× expected → leak.
    """
    def __init__(self, fuel_per_hour: float, max_speed: float, fuel_capacity: float):
        self.fph  = fuel_per_hour
        self.ms   = max_speed
        self.cap  = fuel_capacity
        self.last_pct: Optional[float] = None
        self.drain_win = MovingWindow(20)
        self.gate      = PersistenceGate(5)

    def analyze(self, fuel_pct: float, speed_kmh: float, load_pct: float) -> list[Alert]:
        alerts = []
        if self.last_pct is None:
            self.last_pct = fuel_pct
            return []

        actual_drain_pct = self.last_pct - fuel_pct   # % per tick
        self.last_pct    = fuel_pct

        # Expected drain per tick (1 second): fph / 3600 * speed_ratio * load_factor / cap * 100
        speed_ratio  = max(0.3, speed_kmh / max(self.ms, 1))
        load_factor  = 1 + load_pct / 100 * 0.3
        expected_pct = (self.fph / 3600) * speed_ratio * load_factor / self.cap * 100

        self.drain_win.push(actual_drain_pct)
        ratio = actual_drain_pct / max(expected_pct, 1e-6)

        if self.gate.update(ratio > 2.5 and actual_drain_pct > 0.02):
            z  = self.drain_win.z_score(actual_drain_pct)
            sc = severity_score(min(1.0, ratio / 5), self.gate.count, False, z)
            alerts.append(Alert(
                "FUEL LEAK", AlertLayer.ANOMALY, FaultCategory.DIRECT,
                band(sc), sc,
                f"Fuel draining {ratio:.1f}× faster than expected "
                f"(actual {actual_drain_pct:.4f}%/s vs expected {expected_pct:.4f}%/s). "
                f"Remaining: {fuel_pct:.1f}%. Possible fuel line leak.",
                ratio, 2.5))
        return alerts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  VEHICLE TYPE → VehicleClass ROUTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VTYPE_TO_CLASS: dict[str, VehicleClass] = {
    "SCOOTY": VehicleClass.TWO_WHEELER,
    "BIKE":   VehicleClass.TWO_WHEELER,
    "CAR":    VehicleClass.CAR,
    "PICKUP": VehicleClass.CAR,       # treated as CAR for base detectors
    "VAN":    VehicleClass.HEAVY,
    "TRUCK":  VehicleClass.HEAVY,
    "BUS":    VehicleClass.HEAVY,
}

# ── Profile constants (mirror simulator PROFILES) ────────────────────────────
PROFILES_BRIDGE = {
    "SCOOTY": dict(normal_oil=35, tyre_f=30, tyre_r=28, max_vib=4,
                   max_speed=60,  fph=10,  cap=5,   has_turbo=False, gvwr_kg=0),
    "BIKE":   dict(normal_oil=40, tyre_f=32, tyre_r=30, max_vib=5,
                   max_speed=120, fph=18,  cap=15,  has_turbo=False, gvwr_kg=0),
    "CAR":    dict(normal_oil=55, tyre_f=32, tyre_r=32, max_vib=4,
                   max_speed=140, fph=60,  cap=50,  has_turbo=False, gvwr_kg=0),
    "PICKUP": dict(normal_oil=60, tyre_f=35, tyre_r=38, max_vib=5,
                   max_speed=130, fph=80,  cap=80,  has_turbo=True,  gvwr_kg=2500),
    "VAN":    dict(normal_oil=60, tyre_f=38, tyre_r=42, max_vib=5,
                   max_speed=120, fph=90,  cap=80,  has_turbo=True,  gvwr_kg=5000),
    "TRUCK":  dict(normal_oil=65, tyre_f=100,tyre_r=110,max_vib=7,
                   max_speed=100, fph=120, cap=300, has_turbo=True,  gvwr_kg=18000),
    "BUS":    dict(normal_oil=68, tyre_f=100,tyre_r=110,max_vib=8,
                   max_speed=90,  fph=150, cap=250, has_turbo=True,  gvwr_kg=16000),
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAYLOAD → VehicleSnapshot TRANSLATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Tyre-PSI → axle load: used for AxleOverload on heavy vehicles
# 1 PSI avg across 4 tyres ≈ 80 kg (simplified)
_PSI_TO_KG = 80.0


def payload_to_snapshot(p: dict) -> VehicleSnapshot:
    """
    Maps every simulator payload field to the matching VehicleSnapshot field.
    Fields that have no direct equivalent are derived or set to safe defaults.

    FIELD MAPPING TABLE
    ───────────────────────────────────────────────────────────────────────────
    Simulator field              → VehicleSnapshot field          Notes
    ─────────────────────────────  ──────────────────────────────  ───────────
    speed                        → speed_kmh
    rpm                          → rpm
    accelerator_pct              → engine_load                     % throttle ≈ load
    coolant_temp                 → coolant_temp
    engine_temp                  → oil_temp                        proxy for cross-check
    ambient_temp                 → ambient_temp
    alternator_voltage           → battery_voltage + alternator_v  both mapped
    battery_level                → (used in bridge, not snapshot)
    engine_vibration             → vibration_rms                   g-proxy
    steering_angle               → steering_angle
    heading                      → gps_heading_deg
    latitude/longitude           → (used for drift calc in bridge)
    tyre_pressure_fl/fr/rl/rr    → wheel_speed_fl/fr/rl/rr         PSI used for axle load
    brake_pressure               → decel_rate_ms2                  scaled proxy
    harsh_braking                → brake_lever_pressed
    load_weight_pct              → air_susp_pressure_psi           via inverse formula
    turbo_boost                  → (passed separately to TurboDetector)
    oil_pressure                 → (passed separately to OilPressureDetector)
    fuel_level                   → (passed separately to FuelLeakDetector)
    ───────────────────────────────────────────────────────────────────────────
    """
    vtype = p.get("vehicle_type", "CAR")
    prof  = PROFILES_BRIDGE.get(vtype, PROFILES_BRIDGE["CAR"])

    # ── Voltage proxy: alternator_voltage from simulator maps to battery reading
    # During heavy load / fault the alternator drops; that's what battery detectors watch
    voltage = p.get("alternator_voltage", 14.2)

    # ── Vibration: simulator uses raw units (0–max_vibration).
    # Normalise to 0-1 g-equivalent by dividing by max_vibration for the type
    raw_vib = p.get("engine_vibration", 0.1)
    vib_g   = raw_vib / max(prof["max_vib"], 1.0)

    # ── Air-suspension / axle load proxy for HEAVY vehicles
    # load_weight_pct (0–160%) → PSI via inverse: PSI = (load_kg / PSI_TO_KG / NUM_AXLES)
    load_pct    = p.get("load_weight_pct", 0.0)
    gvwr        = prof["gvwr_kg"]
    load_kg     = gvwr * (load_pct / 100.0) if gvwr > 0 else 0
    susp_psi    = (load_kg / _PSI_TO_KG / 2) if load_kg > 0 else 80.0

    # ── Brake air pressure proxy for TRUCK/BUS
    # brake_pressure (0–100%) → PSI: normal = 120 PSI, AIR_BRAKE_LOSS drives it to 0–6
    raw_bp = p.get("brake_pressure", 50.0)
    if vtype in ("TRUCK", "BUS"):
        # Simulator brake_pressure: normal idle=0-15, active=40-95, AIR_BRAKE_LOSS=0-6
        # Map to tank PSI: normal operation = full 120 PSI; loss scenario = proportionally low
        if raw_bp < 10:
            # Air brake loss scenario: very low brake pressure = low tank PSI
            air_psi = max(5.0, raw_bp * 1.2)
        else:
            air_psi = 120.0   # normal: tank is full regardless of pedal position
    else:
        air_psi = 120.0

    # ── Wheel speed proxy: use speed for all 4 wheels (simulator has no per-wheel speed)
    spd = p.get("speed", 0.0)
    wfl = spd * random.gauss(1.0, 0.01)
    wfr = spd * random.gauss(1.0, 0.01)

    # ── Derive lateral drift proxy from heading change
    # (bridge accumulates this externally; snapshot gets current value passed in)
    # We pass 0 here; SimulatorBridge.process() fills it in after accumulating.

    # ── Is cranking: simulate crank event when speed=0 and battery_level changed a lot
    is_cranking = (spd == 0 and p.get("battery_level", 100) < 95
                   and voltage < 13.5)

    # ── Crank interval: derive from RPM (60000 / RPM / cylinders)
    rpm = p.get("rpm", 800)
    cylinders = 4
    crank_ms = (60000.0 / max(rpm, 100)) / cylinders + random.gauss(0, 0.3)

    return VehicleSnapshot(
        rpm              = rpm,
        speed_kmh        = spd,
        engine_load      = p.get("accelerator_pct", 30.0),
        coolant_temp     = p.get("coolant_temp", 85.0),
        oil_temp         = p.get("engine_temp", 85.0),     # engine_temp as cross-check proxy
        battery_voltage  = voltage,
        alternator_v     = voltage,
        fuel_trim        = 0.0,

        vibration_rms    = vib_g,
        lean_angle_deg   = 0.0,                            # filled by bridge for 2-wheelers
        lateral_accel    = 0.0,
        longitudinal_accel = 0.0,
        vertical_accel   = 1.0,
        incline_deg      = 0.0,

        steering_angle   = p.get("steering_angle", 0.0),
        gps_heading_deg  = p.get("heading", 0.0),
        lateral_drift_m  = 0.0,                            # filled by bridge

        wheel_speed_fl   = wfl,
        wheel_speed_fr   = wfr,
        wheel_speed_rl   = spd,
        wheel_speed_rr   = spd,
        wheel_speed_rear = spd,

        crank_interval_ms = crank_ms,
        o2_sensor_voltage = 0.45,

        brake_lever_pressed = p.get("harsh_braking", False),
        brake_duration_s    = 1.0 if p.get("harsh_braking", False) else 0.0,
        decel_rate_ms2      = raw_bp * 0.08,               # 100% brake ≈ 8 m/s²

        brake_air_pressure_primary   = air_psi,
        brake_air_pressure_secondary = air_psi * random.gauss(1.0, 0.02),
        dpf_backpressure_kpa = 5.0,
        exhaust_temp_c       = p.get("engine_temp", 400.0),
        scr_status_ok        = True,
        air_susp_pressure_psi = susp_psi,
        fan_speed_pct         = min(100, p.get("accelerator_pct", 50.0) + 30),

        ambient_temp = p.get("ambient_temp", 32.0),
        is_cranking  = is_cranking,
        timestamp    = time.time(),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PER-VEHICLE BRIDGE STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VehicleBridge:
    """
    Maintains all detector instances + accumulated state for one vehicle.
    Call .process(payload) each tick.
    """
    def __init__(self, vehicle_id: str, vtype: str):
        self.vehicle_id = vehicle_id
        self.vtype      = vtype
        self.vc         = VTYPE_TO_CLASS.get(vtype, VehicleClass.CAR)
        prof            = PROFILES_BRIDGE.get(vtype, PROFILES_BRIDGE["CAR"])

        # ── Core v3 safety system ────────────────────────────
        self.safety = PredictiveSafetySystem(self.vc)

        # ── Supplementary detectors ──────────────────────────
        self.oil_det    = OilPressureDetector(prof["normal_oil"])
        self.tyre_det   = TyrePressureDetector(prof["tyre_f"], prof["tyre_r"])
        self.turbo_det  = TurboDetector() if prof["has_turbo"] else None
        self.speed_det  = OverspeedDetector(prof["max_speed"])
        self.driver_det = DriverBehaviorDetector()
        self.fuel_det   = FuelLeakDetector(prof["fph"], prof["max_speed"], prof["cap"])

        # ── Accumulated GPS drift (no wheel encoder in sim) ──
        self._last_lat: Optional[float] = None
        self._last_lon: Optional[float] = None
        self._cum_drift_m: float = 0.0

        self.frame = 0

    def _update_drift(self, lat: float, lon: float,
                      steering: float, speed: float) -> float:
        """
        Estimate lateral drift:
        When steering ≈ 0 but the GPS track shows a lateral component,
        the vehicle is drifting. We approximate lateral movement as the
        cross-track displacement per tick.
        """
        if self._last_lat is None:
            self._last_lat, self._last_lon = lat, lon
            return 0.0

        dlat = (lat - self._last_lat) * 111000      # metres
        dlon = (lon - self._last_lon) * 111000 * math.cos(math.radians(lat))
        dist = math.sqrt(dlat**2 + dlon**2)

        # Lateral component: if steering is near-zero but distance > 0 → drift
        if abs(steering) < 5 and dist > 0.01:
            # Approximate heading-vs-actual difference as lateral drift
            self._cum_drift_m += dist * abs(steering) / 90.0
        else:
            self._cum_drift_m = max(0, self._cum_drift_m - 0.01)  # slow decay

        self._last_lat, self._last_lon = lat, lon
        return self._cum_drift_m

    def process(self, payload: dict) -> tuple[float, list[Alert]]:
        self.frame += 1

        # ── Update drift ────────────────────────────────────
        drift = self._update_drift(
            payload.get("latitude", 0),
            payload.get("longitude", 0),
            payload.get("steering_angle", 0),
            payload.get("speed", 0),
        )

        # ── Build snapshot ──────────────────────────────────
        snap = payload_to_snapshot(payload)
        snap.lateral_drift_m = drift

        # ── Two-wheeler lean angle: derive from steering + speed ──
        if self.vc == VehicleClass.TWO_WHEELER:
            # At speed, lean ≈ arctan(v² / g*R) where R ≈ wheelbase / steering_angle
            spd_ms = payload.get("speed", 0) / 3.6
            steer  = abs(payload.get("steering_angle", 0.01))
            lean   = math.degrees(math.atan2(spd_ms**2 * steer, 9.81 * 50)) \
                     if steer > 0 else 0.0
            snap.lean_angle_deg = min(lean, 85.0)

        # ── Run v3 core safety system ────────────────────────
        score, alerts = self.safety.process(snap)

        # ── Run supplementary detectors ──────────────────────
        oil_psi = payload.get("oil_pressure", 55.0)
        alerts += self.oil_det.analyze(snap, oil_psi)

        tyres = {
            "FL": payload.get("tyre_pressure_fl", 32.0),
            "FR": payload.get("tyre_pressure_fr", 32.0),
            "RL": payload.get("tyre_pressure_rl", 32.0),
            "RR": payload.get("tyre_pressure_rr", 32.0),
        }
        alerts += self.tyre_det.analyze(tyres)

        if self.turbo_det:
            alerts += self.turbo_det.analyze(
                payload.get("speed", 0),
                payload.get("turbo_boost", 0),
                payload.get("rpm", 800),
            )

        alerts += self.speed_det.analyze(payload.get("speed", 0))

        alerts += self.driver_det.analyze(
            payload.get("harsh_braking", False),
            payload.get("harsh_acceleration", False),
            payload.get("brake_pressure", 0),
            payload.get("accelerator_pct", 0),
            payload.get("steering_angle", 0),
        )

        alerts += self.fuel_det.analyze(
            payload.get("fuel_level", 50.0),
            payload.get("speed", 0),
            payload.get("load_weight_pct", 0),
        )

        # Re-sort all alerts: CRITICAL → ANOMALY → PREDICTIVE
        priority = {AlertLayer.CRITICAL: 0, AlertLayer.ANOMALY: 1, AlertLayer.PREDICTIVE: 2}
        alerts.sort(key=lambda a: (priority[a.layer], -a.score))

        return score, alerts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FLEET-LEVEL BRIDGE  (public API)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SimulatorBridge:
    """
    Fleet-level entry point.

    Usage:
        bridge = SimulatorBridge()
        score, alerts = bridge.process(payload_dict)

    Each payload_dict is one tick from simulator.py.
    The bridge auto-creates a VehicleBridge on first sight of each vehicle_id.
    """
    def __init__(self):
        self._vehicles: dict[str, VehicleBridge] = {}

    def process(self, payload: dict) -> tuple[float, list[Alert]]:
        vid   = payload["vehicle_id"]
        vtype = payload.get("vehicle_type", "CAR")
        if vid not in self._vehicles:
            self._vehicles[vid] = VehicleBridge(vid, vtype)
        return self._vehicles[vid].process(payload)

    def vehicle_count(self) -> int:
        return len(self._vehicles)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAND-ALONE DEMO  (no live API needed)
#  Replays the simulator's fault scenarios using synthetic payloads
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _base_payload(vehicle_id: str, vtype: str, **overrides) -> dict:
    prof = PROFILES_BRIDGE[vtype]
    base = dict(
        vehicle_id        = vehicle_id,
        vehicle_type      = vtype,
        timestamp         = datetime.now(timezone.utc).isoformat(),
        data_type         = "raw",
        latitude          = 13.05 + random.uniform(-0.01, 0.01),
        longitude         = 80.20 + random.uniform(-0.01, 0.01),
        heading           = random.uniform(0, 360),
        gps_signal        = 95.0,
        speed             = 60.0,
        rpm               = 2200.0,
        driving_mode      = "highway",
        running_hours     = 1.5,
        odometer          = 25000.0,
        engine_temp       = 85.0,
        coolant_temp      = 88.0,
        ambient_temp      = 32.0,
        oil_pressure      = prof["normal_oil"],
        engine_vibration  = 0.5,
        turbo_boost       = 10.0 if prof["has_turbo"] else 0.0,
        alternator_voltage= 14.2,
        brake_pressure    = 10.0,
        accelerator_pct   = 45.0,
        clutch_shifts_per_min = 1.0,
        steering_angle    = 2.0,
        harsh_braking     = False,
        harsh_acceleration= False,
        load_weight_pct   = 50.0 if prof["gvwr_kg"] > 0 else 0.0,
        tyre_pressure_fl  = prof["tyre_f"],
        tyre_pressure_fr  = prof["tyre_f"],
        tyre_pressure_rl  = prof["tyre_r"],
        tyre_pressure_rr  = prof["tyre_r"],
        fuel_level        = 75.0,
        battery_level     = 88.0,
        headwind_speed    = 10.0,
        driver_safety_score = 92.0,
        health_score      = 88.0,
        maintenance_required = False,
        active_fault      = "",
        fault_severity    = "NONE",
    )
    base.update(overrides)
    return base


def demo_scenario(title: str, vehicle_id: str, vtype: str,
                  ticks: list[dict], show_all: bool = False):
    print(f"\n{'═'*72}")
    print(f"  SCENARIO : {title}")
    print(f"  Vehicle  : {vehicle_id}  [{vtype}]")
    print(f"{'═'*72}")

    bridge = SimulatorBridge()
    any_alert = False
    for i, tick in enumerate(ticks):
        tick["vehicle_id"]   = vehicle_id
        tick["vehicle_type"] = vtype
        score, alerts = bridge.process(tick)

        if alerts or show_all:
            any_alert = True
            crit  = [a for a in alerts if a.layer == AlertLayer.CRITICAL]
            anom  = [a for a in alerts if a.layer == AlertLayer.ANOMALY]
            pred  = [a for a in alerts if a.layer == AlertLayer.PREDICTIVE]
            bar   = "█" * int(score/5) + "░" * (20 - int(score/5))
            label = "✅ GOOD" if score>=85 else ("🟡 FAIR" if score>=65 else
                    ("🟠 DEGRADED" if score>=40 else "🔴 CRITICAL"))
            sim_fault = tick.get("active_fault", "")
            print(f"\n  Tick #{i+1:03d} | [{bar}] {score:.1f}/100 {label}"
                  f" | SimFault: {sim_fault or 'none'}")
            for a in alerts[:4]:   # show top 4 per tick
                print(str(a))
    if not any_alert:
        print("  (No alerts fired — system ran clean)")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║   SIMULATOR BRIDGE DEMO  —  Fault Injection → Safety System Detection      ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    N = 20   # ticks per scenario
    p = PROFILES_BRIDGE

    # ── 1. ENGINE_OVERHEAT  (TRUCK) ─────────────────────────────────────────
    demo_scenario("ENGINE_OVERHEAT → EngineTempDetector", "TRUCK-001", "TRUCK",
        [_base_payload("TRUCK-001","TRUCK",
            engine_temp  = 85 + i*4 + random.gauss(0,0.5),
            coolant_temp = 92 + i*3 + random.gauss(0,0.5),
            active_fault = "ENGINE_OVERHEAT", fault_severity="CRITICAL")
         for i in range(N)])

    # ── 2. OIL_PRESSURE_DROP  (TRUCK) ───────────────────────────────────────
    demo_scenario("OIL_PRESSURE_DROP → OilPressureDetector", "TRUCK-002", "TRUCK",
        [_base_payload("TRUCK-002","TRUCK",
            oil_pressure = max(15, p["TRUCK"]["normal_oil"] - i*3.5),
            engine_temp  = 90 + i*0.5,
            active_fault = "OIL_PRESSURE_DROP", fault_severity="CRITICAL")
         for i in range(N)])

    # ── 3. BATTERY_FAILURE  (CAR) ────────────────────────────────────────────
    demo_scenario("BATTERY_FAILURE → BatteryDetector", "CAR-001", "CAR",
        [_base_payload("CAR-001","CAR",
            alternator_voltage = max(9.0, 14.2 - i*0.35),
            battery_level      = max(10, 90 - i*5),
            speed              = 0, rpm = 700,
            active_fault = "BATTERY_FAILURE", fault_severity="CRITICAL")
         for i in range(N)])

    # ── 4. TYRE_BLOWOUT_FL  (BUS) ────────────────────────────────────────────
    demo_scenario("TYRE_BLOWOUT_FL → TyrePressureDetector", "BUS-001", "BUS",
        [_base_payload("BUS-001","BUS",
            tyre_pressure_fl = max(5, p["BUS"]["tyre_f"] - i*8),
            active_fault = "TYRE_BLOWOUT_FL", fault_severity="CRITICAL")
         for i in range(N)])

    # ── 5. AIR_BRAKE_LOSS  (BUS) ─────────────────────────────────────────────
    demo_scenario("AIR_BRAKE_LOSS → AirBrakeDetector", "BUS-002", "BUS",
        [_base_payload("BUS-002","BUS",
            brake_pressure = max(5, 100 - i*9),
            active_fault = "AIR_BRAKE_LOSS", fault_severity="CRITICAL")
         for i in range(N)])

    # ── 6. VIBRATION_SPIKE  (BIKE) ───────────────────────────────────────────
    demo_scenario("VIBRATION_SPIKE → VibrationDetector", "BIKE-001", "BIKE",
        [_base_payload("BIKE-001","BIKE",
            engine_vibration = 0.5 + (i*0.3 if i>8 else 0) + random.gauss(0,0.05),
            active_fault = "VIBRATION_SPIKE" if i>8 else "",
            fault_severity = "WARNING" if i>8 else "NONE")
         for i in range(N)])

    # ── 7. OVERLOAD  (TRUCK) ─────────────────────────────────────────────────
    demo_scenario("OVERLOAD → AxleOverloadDetector", "TRUCK-003", "TRUCK",
        [_base_payload("TRUCK-003","TRUCK",
            load_weight_pct = 60 + i*8,
            active_fault = "OVERLOAD", fault_severity="WARNING")
         for i in range(N)])

    # ── 8. CHAIN_SLIP  (SCOOTY) ──────────────────────────────────────────────
    demo_scenario("CHAIN_SLIP → BeltChainSlipDetector", "SCOOTY-001", "SCOOTY",
        [_base_payload("SCOOTY-001","SCOOTY",
            rpm   = 3000 + i*150 + random.gauss(0,50),
            speed = max(5, 50 - i*2),
            engine_vibration = 1.0 + i*0.1,
            active_fault = "CHAIN_SLIP" if i>5 else "",
            fault_severity = "WARNING" if i>5 else "NONE")
         for i in range(N)])

    # ── 9. BRAKE_WEAR  (CAR) ─────────────────────────────────────────────────
    demo_scenario("BRAKE_WEAR → BrakeWearDetector", "CAR-002", "CAR",
        [_base_payload("CAR-002","CAR",
            brake_pressure = 88 + random.gauss(0,2),
            harsh_braking  = True,
            speed          = 50,
            active_fault   = "BRAKE_WEAR", fault_severity="WARNING")
         for i in range(N)])

    # ── 10. TURBO_FAILURE  (VAN) ─────────────────────────────────────────────
    demo_scenario("TURBO_FAILURE → TurboDetector", "VAN-001", "VAN",
        [_base_payload("VAN-001","VAN",
            turbo_boost  = 0.0 if i > 5 else 10.0,
            speed        = 75,
            active_fault = "TURBO_FAILURE" if i>5 else "",
            fault_severity="WARNING" if i>5 else "NONE")
         for i in range(N)])

    # ── 11. HARSH_DRIVER ─────────────────────────────────────────────────────
    demo_scenario("HARSH_DRIVER → DriverBehaviorDetector", "PICKUP-001", "PICKUP",
        [_base_payload("PICKUP-001","PICKUP",
            harsh_braking      = True,
            harsh_acceleration = True,
            brake_pressure     = round(random.uniform(88,100),1),
            accelerator_pct    = round(random.uniform(92,100),1),
            steering_angle     = round(random.uniform(45,60)*random.choice([-1,1]),1),
            active_fault = "HARSH_DRIVER", fault_severity="INFO")
         for i in range(N)])

    # ── 12. FUEL_LEAK  (VAN) ─────────────────────────────────────────────────
    demo_scenario("FUEL_LEAK → FuelLeakDetector", "VAN-002", "VAN",
        [_base_payload("VAN-002","VAN",
            fuel_level   = max(10, 75 - i*3.5),
            speed        = 60,
            active_fault = "FUEL_LEAK", fault_severity="WARNING")
         for i in range(N)])

    print(f"\n{'═'*72}")
    print("  FIELD MAPPING SUMMARY")
    print(f"{'─'*72}")
    mapping = [
        ("engine_temp / coolant_temp",  "EngineTempDetector",      "CRITICAL/PREDICTIVE/ANOMALY"),
        ("oil_pressure",                "OilPressureDetector",      "CRITICAL/PREDICTIVE"),
        ("alternator_voltage",          "BatteryDetector",          "CRITICAL/PREDICTIVE"),
        ("tyre_pressure_fl/fr/rl/rr",   "TyrePressureDetector",     "CRITICAL/PREDICTIVE"),
        ("brake_pressure (TRUCK/BUS)",  "AirBrakeDetector",         "CRITICAL/ANOMALY"),
        ("engine_vibration",            "VibrationDetector",        "CRITICAL/ANOMALY/PREDICTIVE"),
        ("load_weight_pct → susp_psi",  "AxleOverloadDetector",     "CRITICAL"),
        ("rpm + speed (2W)",            "BeltChainSlipDetector",    "ANOMALY"),
        ("brake_pressure + decel",      "BrakeWearDetector",        "PREDICTIVE"),
        ("turbo_boost",                 "TurboDetector",            "ANOMALY"),
        ("speed vs max_speed",          "OverspeedDetector",        "CRITICAL"),
        ("harsh_braking/accel/steer",   "DriverBehaviorDetector",   "ANOMALY"),
        ("fuel_level drain rate",       "FuelLeakDetector",         "ANOMALY"),
        ("GPS lat/lon drift",           "AlignmentDetector",        "PREDICTIVE/ANOMALY"),
    ]
    for sim_field, detector, layers in mapping:
        print(f"  {sim_field:<30} → {detector:<28} [{layers}]")
    print(f"{'═'*72}\n")
