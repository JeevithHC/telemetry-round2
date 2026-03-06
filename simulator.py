"""
simulator.py — Realistic Vehicle Telemetry Simulator v2
=========================================================
Fleet : SCOOTY x10, BIKE x10, CAR x20, PICKUP x10,
        VAN x15, TRUCK x20, BUS x15  =  100 vehicles
Data  : 35+ fields per second per vehicle
Physics: speed drives RPM and temperature, fuel drains
         with load, tyres slowly leak, oil degrades

Fault injection: each vehicle randomly enters a fault state
  for a burst of ticks, pushing specific fields past alert
  thresholds. Faults are vehicle-type aware (e.g. OVERLOAD
  only on load-capable types, TURBO_FAILURE only on turbo
  vehicles). Payload carries active_fault + fault_severity
  so the backend can detect and alert without re-deriving.

Windows: runs as-is — no changes needed.
         Start with:  python simulator.py
"""

import requests
import random
import time
import threading
import math
from datetime import datetime, timezone

API_URL       = "http://localhost:8000/api/telemetry"
SEND_INTERVAL = 1  # seconds between sends per vehicle

# ── Fleet Definition ───────────────────────────────────────
VEHICLES = (
    [f"SCOOTY-{i:03d}" for i in range(1, 11)] +
    [f"BIKE-{i:03d}"   for i in range(1, 11)] +
    [f"CAR-{i:03d}"    for i in range(1, 21)] +
    [f"PICKUP-{i:03d}" for i in range(1, 11)] +
    [f"VAN-{i:03d}"    for i in range(1, 16)] +
    [f"TRUCK-{i:03d}"  for i in range(1, 21)] +
    [f"BUS-{i:03d}"    for i in range(1, 16)]
)
random.shuffle(VEHICLES)

# ── Chennai GPS Bounds ─────────────────────────────────────
GPS_BOUNDS = {
    "lat_min": 12.92, "lat_max": 13.18,
    "lon_min": 80.12, "lon_max": 80.27
}

# ── Vehicle Profiles ───────────────────────────────────────
PROFILES = {
    "SCOOTY": {
        "max_speed": 60,    "idle_rpm": 1200, "max_rpm": 7000,
        "base_temp": 55,    "max_temp": 95,   "temp_per_hour": 4,  "temp_per_speed": 0.07,
        "fuel_capacity": 5, "fuel_per_hour": 10, "battery_drain": 0.020,
        "has_turbo": False, "has_clutch": True, "max_clutch_shifts": 8,
        "load_capable": False, "two_wheeler": True,
        "normal_oil_psi": 35, "normal_coolant": 65,
        "tyre_psi_front": 30, "tyre_psi_rear": 28,
        "max_vibration": 4,   "normal_voltage": 13.8, "max_load_kg": 0,
    },
    "BIKE": {
        "max_speed": 120,   "idle_rpm": 1000, "max_rpm": 9000,
        "base_temp": 60,    "max_temp": 100,  "temp_per_hour": 5,  "temp_per_speed": 0.09,
        "fuel_capacity": 15,"fuel_per_hour": 18, "battery_drain": 0.020,
        "has_turbo": False, "has_clutch": True, "max_clutch_shifts": 12,
        "load_capable": False, "two_wheeler": True,
        "normal_oil_psi": 40, "normal_coolant": 70,
        "tyre_psi_front": 32, "tyre_psi_rear": 30,
        "max_vibration": 5,   "normal_voltage": 13.8, "max_load_kg": 0,
    },
    "CAR": {
        "max_speed": 140,   "idle_rpm": 700,  "max_rpm": 5000,
        "base_temp": 70,    "max_temp": 105,  "temp_per_hour": 6,  "temp_per_speed": 0.10,
        "fuel_capacity": 50,"fuel_per_hour": 60, "battery_drain": 0.015,
        "has_turbo": False, "has_clutch": True, "max_clutch_shifts": 6,
        "load_capable": False, "two_wheeler": False,
        "normal_oil_psi": 55, "normal_coolant": 88,
        "tyre_psi_front": 32, "tyre_psi_rear": 32,
        "max_vibration": 4,   "normal_voltage": 14.2, "max_load_kg": 0,
    },
    "PICKUP": {
        "max_speed": 130,   "idle_rpm": 750,  "max_rpm": 4000,
        "base_temp": 75,    "max_temp": 110,  "temp_per_hour": 7,  "temp_per_speed": 0.12,
        "fuel_capacity": 80,"fuel_per_hour": 80, "battery_drain": 0.015,
        "has_turbo": True,  "has_clutch": True, "max_clutch_shifts": 7,
        "load_capable": True, "two_wheeler": False,
        "normal_oil_psi": 60, "normal_coolant": 90,
        "tyre_psi_front": 35, "tyre_psi_rear": 38,
        "max_vibration": 5,   "normal_voltage": 14.2, "max_load_kg": 1000,
    },
    "VAN": {
        "max_speed": 120,   "idle_rpm": 750,  "max_rpm": 3500,
        "base_temp": 72,    "max_temp": 110,  "temp_per_hour": 7,  "temp_per_speed": 0.12,
        "fuel_capacity": 80,"fuel_per_hour": 90, "battery_drain": 0.018,
        "has_turbo": True,  "has_clutch": True, "max_clutch_shifts": 6,
        "load_capable": True, "two_wheeler": False,
        "normal_oil_psi": 60, "normal_coolant": 90,
        "tyre_psi_front": 38, "tyre_psi_rear": 42,
        "max_vibration": 5,   "normal_voltage": 14.2, "max_load_kg": 1500,
    },
    "TRUCK": {
        "max_speed": 100,    "idle_rpm": 750,  "max_rpm": 2500,
        "base_temp": 75,     "max_temp": 115,  "temp_per_hour": 8,  "temp_per_speed": 0.15,
        "fuel_capacity": 300,"fuel_per_hour": 120, "battery_drain": 0.020,
        "has_turbo": True,   "has_clutch": True, "max_clutch_shifts": 5,
        "load_capable": True, "two_wheeler": False,
        "normal_oil_psi": 65, "normal_coolant": 92,
        "tyre_psi_front": 100,"tyre_psi_rear": 110,
        "max_vibration": 7,   "normal_voltage": 14.4, "max_load_kg": 10000,
    },
    "BUS": {
        "max_speed": 90,     "idle_rpm": 700,  "max_rpm": 2200,
        "base_temp": 78,     "max_temp": 118,  "temp_per_hour": 9,  "temp_per_speed": 0.16,
        "fuel_capacity": 250,"fuel_per_hour": 150, "battery_drain": 0.025,
        "has_turbo": True,   "has_clutch": True, "max_clutch_shifts": 4,
        "load_capable": True, "two_wheeler": False,
        "normal_oil_psi": 68, "normal_coolant": 93,
        "tyre_psi_front": 100,"tyre_psi_rear": 110,
        "max_vibration": 8,   "normal_voltage": 14.4, "max_load_kg": 8000,
    },
}

# ── Fault catalogue ────────────────────────────────────────
# (duration_lo, duration_hi, severity, applies_to)
# applies_to=None means all vehicle types
FAULT_CATALOGUE = {
    "ENGINE_OVERHEAT":         (15, 45, "CRITICAL", None),
    "OIL_PRESSURE_DROP":       (20, 60, "CRITICAL", None),
    "BATTERY_FAILURE":         (15, 40, "CRITICAL", None),
    "TYRE_BLOWOUT_FL":         (10, 25, "CRITICAL", None),
    "TYRE_BLOWOUT_RR":         (10, 25, "CRITICAL", None),
    "COOLANT_LEAK":            (20, 70, "WARNING",  None),
    "VIBRATION_SPIKE":         (10, 25, "WARNING",  None),
    "FUEL_LEAK":               (15, 45, "WARNING",  None),
    "OVERSPEED":               (10, 30, "WARNING",  None),
    "HARSH_DRIVER":            (15, 60, "INFO",     None),
    "GPS_DRIFT":               (8,  20, "INFO",     None),
    # vehicle-specific
    "OVERLOAD":                (20, 60, "WARNING",  {"PICKUP","VAN","TRUCK","BUS"}),
    "TURBO_FAILURE":           (15, 40, "WARNING",  {"PICKUP","VAN","TRUCK","BUS"}),
    "CHAIN_SLIP":              (10, 25, "WARNING",  {"SCOOTY","BIKE"}),
    "BRAKE_WEAR":              (20, 50, "WARNING",  {"CAR","PICKUP","VAN"}),
    "AIR_BRAKE_LOSS":          (10, 25, "CRITICAL", {"TRUCK","BUS"}),
}
FAULT_TRIGGER_PROB = 0.05   # 5% chance per tick to start a new fault


def get_type(vid: str) -> str:
    for t in ["SCOOTY","BIKE","CAR","PICKUP","VAN","TRUCK","BUS"]:
        if t in vid:
            return t
    return "CAR"


def pick_fault(vtype: str):
    """Return a random fault name valid for this vehicle type."""
    candidates = [
        name for name, cfg in FAULT_CATALOGUE.items()
        if cfg[3] is None or vtype in cfg[3]
    ]
    return random.choice(candidates)


def calc_driver_safety_score(harsh_braking, harsh_accel, clutch_shifts,
                              brake_pressure, steering_angle, max_clutch):
    score = 100.0
    if harsh_braking:                      score -= 15
    if harsh_accel:                        score -= 12
    if clutch_shifts > max_clutch * 0.8:  score -= 8
    if brake_pressure > 85:               score -= 5
    if abs(steering_angle) > 35:          score -= 5
    return round(max(0, score), 1)


def calc_health_score(oil_psi, normal_oil, vibration, max_vib,
                      tyre_pressures, normal_tyre_f, normal_tyre_r,
                      odometer, coolant, normal_coolant):
    score = 100.0
    oil_ratio = oil_psi / normal_oil
    if oil_ratio < 0.5:    score -= 30
    elif oil_ratio < 0.7:  score -= 15
    elif oil_ratio < 0.85: score -= 5
    vib_ratio = vibration / max_vib
    if vib_ratio > 0.85:   score -= 20
    elif vib_ratio > 0.7:  score -= 10
    normals = [normal_tyre_f, normal_tyre_f, normal_tyre_r, normal_tyre_r]
    for tp, n in zip(tyre_pressures, normals):
        dev = abs(tp - n) / n
        if dev > 0.2:   score -= 8
        elif dev > 0.1: score -= 3
    if odometer > 150000:   score -= 15
    elif odometer > 100000: score -= 8
    elif odometer > 50000:  score -= 3
    if coolant > normal_coolant * 1.15: score -= 10
    return round(max(0, min(100, score)), 1)


def simulate_vehicle(vehicle_id: str):
    vtype   = get_type(vehicle_id)
    profile = PROFILES[vtype]

    # ── Initial State ──────────────────────────────────────
    running_seconds = 0
    speed           = 0.0
    temperature     = 25.0
    coolant_temp    = 25.0
    fuel_level      = float(profile["fuel_capacity"])
    battery_level   = 100.0
    odometer        = random.uniform(5000, 80000)
    mode            = random.choice(["city","highway","idle"])
    mode_timer      = 0
    mode_duration   = random.randint(30, 120)
    target_speed    = 0.0
    heading         = random.uniform(0, 360)
    lat             = random.uniform(GPS_BOUNDS["lat_min"]+0.05, GPS_BOUNDS["lat_max"]-0.05)
    lon             = random.uniform(GPS_BOUNDS["lon_min"]+0.05, GPS_BOUNDS["lon_max"]-0.05)

    tyre_fl = profile["tyre_psi_front"] - random.uniform(0, 2)
    tyre_fr = profile["tyre_psi_front"] - random.uniform(0, 2)
    tyre_rl = profile["tyre_psi_rear"]  - random.uniform(0, 2)
    tyre_rr = profile["tyre_psi_rear"]  - random.uniform(0, 2)

    oil_pressure    = profile["normal_oil_psi"] - random.uniform(0, 3)
    load_weight_pct = random.uniform(20, 90) if profile["load_capable"] else 0.0

    # ── Fault state ────────────────────────────────────────
    active_fault        = None   # current fault name, or None
    fault_ticks_left    = 0      # ticks remaining for active fault
    oil_drop_acc        = 0.0    # accumulated oil drop for OIL_PRESSURE_DROP ramp
    coolant_rise_acc    = 0.0    # accumulated rise for COOLANT_LEAK ramp

    print(f"[START] {vehicle_id} ({vtype}) | mode={mode} | lat={lat:.4f} lon={lon:.4f}")

    while True:
        running_hours = running_seconds / 3600.0
        max_speed     = profile["max_speed"]
        timestamp     = datetime.now(timezone.utc)

        # ── Mode switching ─────────────────────────────────
        mode_timer += 1
        if mode_timer >= mode_duration:
            mode = random.choices(
                ["city","highway","idle","braking"],
                weights=[40, 35, 15, 10]
            )[0]
            mode_timer    = 0
            mode_duration = random.randint(30, 120)

        # ── Target speed per mode ──────────────────────────
        if mode == "highway":
            if mode_timer == 0:
                target_speed = random.uniform(max_speed * 0.70, max_speed * 0.95)
        elif mode == "city":
            if mode_timer == 0 or mode_timer % 15 == 0:
                target_speed = random.uniform(5, max_speed * 0.45)
        elif mode == "idle":
            target_speed = 0
        elif mode == "braking":
            target_speed = max(0, speed - random.uniform(3, 10))

        # ── Smooth speed transitions ───────────────────────
        if speed < target_speed:
            speed = min(speed + random.uniform(0.5, 2.5), target_speed)
        elif speed > target_speed:
            speed = max(speed - random.uniform(1.0, 4.0), target_speed)
        speed = round(max(0.0, min(speed, max_speed)), 2)

        # ── RPM — correlated with speed ────────────────────
        idle_rpm = profile["idle_rpm"]
        max_rpm  = profile["max_rpm"]
        if speed == 0:
            rpm = round(random.uniform(idle_rpm - 50, idle_rpm + 150), 0)
        else:
            speed_ratio = speed / max_speed
            rpm = idle_rpm + (max_rpm - idle_rpm) * speed_ratio
            rpm = round(rpm + random.uniform(-100, 100), 0)
            rpm = max(idle_rpm, min(rpm, max_rpm))

        # ── Temperature — physics model ────────────────────
        AMBIENT     = 32.0
        hour_heat   = profile["temp_per_hour"]  * running_hours
        speed_heat  = profile["temp_per_speed"] * speed
        load_heat   = (load_weight_pct / 100) * 5 if profile["load_capable"] else 0
        target_temp = AMBIENT + hour_heat + speed_heat + load_heat
        if running_hours > 0.05:
            target_temp += random.uniform(-1.0, 1.0)
        target_temp = min(target_temp, profile["max_temp"])

        if temperature < target_temp:
            temperature += random.uniform(0.1, 0.5)
        elif temperature > target_temp:
            temperature -= random.uniform(0.05, 0.2)
        temperature = round(max(AMBIENT, min(temperature, profile["max_temp"])), 2)

        # ── Coolant temperature ────────────────────────────
        target_coolant = profile["normal_coolant"] * (temperature / (profile["max_temp"] * 0.8))
        target_coolant = min(target_coolant, profile["normal_coolant"] * 1.2)
        if coolant_temp < target_coolant:
            coolant_temp += random.uniform(0.05, 0.3)
        elif coolant_temp > target_coolant:
            coolant_temp -= random.uniform(0.02, 0.15)
        coolant_temp = round(max(AMBIENT, coolant_temp), 2)

        # ── Oil Pressure — degrades with hours + RPM ───────
        oil_drain    = 0.0001 * running_hours + (rpm / max_rpm) * 0.001
        oil_pressure = max(10, oil_pressure - oil_drain + random.uniform(-0.1, 0.15))
        oil_pressure = round(min(oil_pressure, profile["normal_oil_psi"]), 2)

        # ── Engine Vibration ───────────────────────────────
        age_factor   = min(odometer / 100000, 1.0)
        rpm_factor   = abs(rpm - (idle_rpm + (max_rpm - idle_rpm) * 0.5)) / max_rpm
        vibration    = round(age_factor * 2 + rpm_factor * profile["max_vibration"] + random.uniform(0, 0.5), 2)
        vibration    = min(vibration, profile["max_vibration"])

        # ── Turbo Boost ────────────────────────────────────
        if profile["has_turbo"]:
            turbo_boost = round((speed / max_speed) * 18 + random.uniform(-1, 1), 2) if speed > 30 else 0.0
        else:
            turbo_boost = 0.0

        # ── Alternator Voltage ─────────────────────────────
        alternator_voltage = round(
            profile["normal_voltage"] + random.uniform(-0.3, 0.2) - (battery_level < 30) * 0.5, 2
        )

        # ── Fuel consumption ───────────────────────────────
        load_factor  = 1 + (load_weight_pct / 100) * 0.3 if profile["load_capable"] else 1.0
        fuel_per_sec = (profile["fuel_per_hour"] / 3600) * (0.3 + 0.7 * (speed / max_speed)) * load_factor
        fuel_level   = max(0, fuel_level - fuel_per_sec)
        fuel_pct     = round((fuel_level / profile["fuel_capacity"]) * 100, 2)

        # ── Battery drain ──────────────────────────────────
        battery_level = round(max(0, battery_level - profile["battery_drain"]), 3)

        # ── Tyre pressure — slow leak ──────────────────────
        tyre_fl = round(max(10, tyre_fl - random.uniform(0, 0.003)), 3)
        tyre_fr = round(max(10, tyre_fr - random.uniform(0, 0.003)), 3)
        tyre_rl = round(max(10, tyre_rl - random.uniform(0, 0.002)), 3)
        tyre_rr = round(max(10, tyre_rr - random.uniform(0, 0.002)), 3)

        # ── Odometer ───────────────────────────────────────
        odometer = round(odometer + (speed / 3600), 3)

        # ── Driver Behavior ────────────────────────────────
        if speed == 0:
            accel_pct      = 0.0
            brake_pressure = round(random.uniform(0, 5), 1)
        elif mode == "braking":
            accel_pct      = 0.0
            brake_pressure = round(random.uniform(40, 95), 1)
        else:
            accel_pct      = round((speed / max_speed) * 100 + random.uniform(-5, 5), 1)
            accel_pct      = max(0, min(100, accel_pct))
            brake_pressure = round(random.uniform(0, 15), 1)

        if not profile["has_clutch"] or speed == 0:
            clutch_shifts = 0
        elif mode == "city":
            clutch_shifts = round(random.uniform(2, profile["max_clutch_shifts"]), 1)
        elif mode == "highway":
            clutch_shifts = round(random.uniform(0, 1.5), 1)
        else:
            clutch_shifts = round(random.uniform(0, profile["max_clutch_shifts"] * 0.4), 1)

        if mode == "city":
            steering_angle = round(random.uniform(-40, 40), 1)
        elif mode == "highway":
            steering_angle = round(random.uniform(-8, 8), 1)
        else:
            steering_angle = 0.0

        harsh_braking      = brake_pressure > 80
        harsh_acceleration = accel_pct > 88 and mode != "braking"

        # ── GPS Movement ───────────────────────────────────
        heading += random.uniform(-5, 5)
        heading  = heading % 360

        if speed > 0:
            speed_km_per_sec = speed / 3600
            lat += (speed_km_per_sec / 111) * math.cos(math.radians(heading))
            lon += (speed_km_per_sec / (111 * math.cos(math.radians(lat)))) * math.sin(math.radians(heading))

        if lat < GPS_BOUNDS["lat_min"] or lat > GPS_BOUNDS["lat_max"]:
            heading = (180 - heading) % 360
            lat = max(GPS_BOUNDS["lat_min"], min(GPS_BOUNDS["lat_max"], lat))
        if lon < GPS_BOUNDS["lon_min"] or lon > GPS_BOUNDS["lon_max"]:
            heading = (360 - heading) % 360
            lon = max(GPS_BOUNDS["lon_min"], min(GPS_BOUNDS["lon_max"], lon))

        gps_signal     = round(random.uniform(75, 100), 1)
        ambient_temp   = round(AMBIENT + random.uniform(-2, 2), 1)
        headwind_speed = round(random.uniform(0, 30), 1)

        # ══════════════════════════════════════════════════
        # FAULT LIFECYCLE
        # 1. Expire finished fault, reset accumulators
        # 2. Roll for a new fault if none active
        # 3. Override specific fields based on active fault
        # ══════════════════════════════════════════════════

        # 1. Age down / expire
        if active_fault:
            fault_ticks_left -= 1
            if fault_ticks_left <= 0:
                active_fault     = None
                oil_drop_acc     = 0.0
                coolant_rise_acc = 0.0

        # 2. Possibly start a new fault
        if not active_fault and random.random() < FAULT_TRIGGER_PROB:
            active_fault     = pick_fault(vtype)
            lo, hi, _, _     = FAULT_CATALOGUE[active_fault]
            fault_ticks_left = random.randint(lo, hi)
            oil_drop_acc     = 0.0
            coolant_rise_acc = 0.0

        # 3. Apply fault overrides on top of physics values
        fault_severity = "NONE"
        if active_fault:
            fault_severity = FAULT_CATALOGUE[active_fault][2]
            p = profile

            if active_fault == "ENGINE_OVERHEAT":
                temperature  = round(min(p["max_temp"] * 1.20, temperature  + random.uniform(4, 10)), 2)
                coolant_temp = round(min(p["normal_coolant"] * 1.38, coolant_temp + random.uniform(3, 7)), 2)

            elif active_fault == "OIL_PRESSURE_DROP":
                # Gradual ramp — 1–3 psi lost per tick
                oil_drop_acc += random.uniform(1, 3)
                oil_pressure  = round(max(5, p["normal_oil_psi"] - oil_drop_acc), 2)

            elif active_fault == "BATTERY_FAILURE":
                battery_level      = round(max(0,   battery_level      - random.uniform(3, 6)),   2)
                alternator_voltage = round(max(9.0, alternator_voltage - random.uniform(1, 2.5)), 2)

            elif active_fault == "TYRE_BLOWOUT_FL":
                tyre_fl = round(max(2, tyre_fl - random.uniform(6, 15)), 2)

            elif active_fault == "TYRE_BLOWOUT_RR":
                tyre_rr = round(max(2, tyre_rr - random.uniform(6, 15)), 2)

            elif active_fault == "COOLANT_LEAK":
                # Gradual ramp — coolant temp rises 0.5–1.5 °C per tick
                coolant_rise_acc += random.uniform(0.5, 1.5)
                coolant_temp      = round(min(p["normal_coolant"] * 1.40, p["normal_coolant"] + coolant_rise_acc), 2)
                temperature       = round(min(p["max_temp"] * 1.10, temperature + random.uniform(0.5, 2)), 2)

            elif active_fault == "VIBRATION_SPIKE":
                vibration = round(min(p["max_vibration"] * 1.5, vibration + random.uniform(3, p["max_vibration"])), 2)

            elif active_fault == "FUEL_LEAK":
                # Fuel drains 3× faster than normal
                fuel_level = max(0, fuel_level - (fuel_per_sec * 2))
                fuel_pct   = round((fuel_level / p["fuel_capacity"]) * 100, 2)

            elif active_fault == "OVERSPEED":
                speed = round(min(p["max_speed"] * 1.40, speed + random.uniform(20, 35)), 2)
                rpm   = round(min(max_rpm * 1.20,        rpm   + random.uniform(400, 800)), 0)

            elif active_fault == "HARSH_DRIVER":
                harsh_braking      = True
                harsh_acceleration = True
                brake_pressure     = round(random.uniform(88, 100), 1)
                accel_pct          = round(random.uniform(92, 100), 1)
                steering_angle     = round(random.uniform(45, 60) * random.choice([-1, 1]), 1)

            elif active_fault == "GPS_DRIFT":
                lat        = round(lat + random.uniform(-0.02, 0.02), 6)
                lon        = round(lon + random.uniform(-0.02, 0.02), 6)
                gps_signal = round(random.uniform(5, 28), 1)

            elif active_fault == "OVERLOAD":
                load_weight_pct = round(min(160, load_weight_pct + random.uniform(50, 80)), 1)
                temperature     = round(min(p["max_temp"] * 1.10, temperature + random.uniform(2, 5)), 2)
                tyre_rl         = round(max(10, tyre_rl - random.uniform(5, 12)), 2)
                tyre_rr         = round(max(10, tyre_rr - random.uniform(5, 12)), 2)

            elif active_fault == "TURBO_FAILURE":
                turbo_boost = 0.0
                rpm         = round(min(max_rpm * 1.15, rpm + random.uniform(300, 700)), 0)

            elif active_fault == "CHAIN_SLIP":
                # High RPM but very low speed — chain not engaging
                rpm   = round(min(max_rpm * 0.90, rpm + random.uniform(800, 1800)), 0)
                speed = round(max(0, speed - random.uniform(8, 20)), 2)
                vibration = round(min(p["max_vibration"] * 1.3, vibration + random.uniform(2, 4)), 2)

            elif active_fault == "BRAKE_WEAR":
                brake_pressure = round(random.uniform(88, 100), 1)
                harsh_braking  = True

            elif active_fault == "AIR_BRAKE_LOSS":
                brake_pressure = round(random.uniform(0, 6), 1)
                harsh_braking  = True

        # ── Derived Scores (recalculated after fault overrides) ─
        driver_safety_score = calc_driver_safety_score(
            harsh_braking, harsh_acceleration, clutch_shifts,
            brake_pressure, steering_angle, profile["max_clutch_shifts"]
        )
        health_score = calc_health_score(
            oil_pressure, profile["normal_oil_psi"],
            vibration, profile["max_vibration"],
            [tyre_fl, tyre_fr, tyre_rl, tyre_rr],
            profile["tyre_psi_front"], profile["tyre_psi_rear"],
            odometer, coolant_temp, profile["normal_coolant"]
        )
        maintenance_required = health_score < 40

        # ── Build Payload ──────────────────────────────────
        payload = {
            "vehicle_id":   vehicle_id,
            "vehicle_type": vtype,
            "timestamp":    timestamp.isoformat(),
            "data_type":    "fault" if active_fault else "raw",
            # Location
            "latitude":          round(lat, 6),
            "longitude":         round(lon, 6),
            "heading":           round(heading, 1),
            "gps_signal":        gps_signal,
            # Core
            "speed":             speed,
            "rpm":               rpm,
            "driving_mode":      mode,
            "running_hours":     round(running_hours, 4),
            "odometer":          round(odometer, 1),
            # Thermal
            "engine_temp":       temperature,
            "coolant_temp":      coolant_temp,
            "ambient_temp":      ambient_temp,
            # Engine Health
            "oil_pressure":      round(oil_pressure, 2),
            "engine_vibration":  round(vibration, 2),
            "turbo_boost":       turbo_boost,
            "alternator_voltage":alternator_voltage,
            # Driver
            "brake_pressure":        brake_pressure,
            "accelerator_pct":       accel_pct,
            "clutch_shifts_per_min": clutch_shifts,
            "steering_angle":        steering_angle,
            "harsh_braking":         harsh_braking,
            "harsh_acceleration":    harsh_acceleration,
            # Load & Wear
            "load_weight_pct":   round(load_weight_pct, 1),
            "tyre_pressure_fl":  tyre_fl,
            "tyre_pressure_fr":  tyre_fr,
            "tyre_pressure_rl":  tyre_rl,
            "tyre_pressure_rr":  tyre_rr,
            # Resources
            "fuel_level":        fuel_pct,
            "battery_level":     round(battery_level, 2),
            # Environment
            "headwind_speed":    headwind_speed,
            # Derived
            "driver_safety_score": driver_safety_score,
            "health_score":        health_score,
            "maintenance_required":maintenance_required,
            # Fault metadata — backend keys off these for alerts
            "active_fault":    active_fault or "",
            "fault_severity":  fault_severity,
        }

        # ── Send to API ────────────────────────────────────
        try:
            r = requests.post(API_URL, json=payload, timeout=5)
            if r.status_code == 201:
                maint      = " MAINT!"           if maintenance_required else ""
                fault_str  = f" | FAULT={active_fault}" if active_fault  else ""
                print(
                    f"[{vehicle_id}] {running_hours:.2f}h | "
                    f"{speed}km/h | {rpm:.0f}rpm | "
                    f"eng={temperature}C | fuel={fuel_pct}% | "
                    f"health={health_score}{maint}{fault_str}"
                )
            else:
                print(f"[WARN] {vehicle_id} -> {r.status_code}")
        except Exception as e:
            print(f"[ERROR] {vehicle_id} -> {e}")

        running_seconds += SEND_INTERVAL
        time.sleep(SEND_INTERVAL)


def main():
    print("=" * 65)
    print("   Vehicle Telemetry Simulator v2 - Full Physics Model")
    print(f"   {len(VEHICLES)} vehicles | GPS: Chennai | {SEND_INTERVAL}s interval")
    print(f"   Fault trigger prob: {FAULT_TRIGGER_PROB*100:.0f}% per tick | 16 fault types")
    print("=" * 65)
    print("Press CTRL+C to stop.\n")

    for vid in VEHICLES:
        t = threading.Thread(target=simulate_vehicle, args=(vid,), daemon=True)
        t.start()
        time.sleep(0.05)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[STOP] Simulator shut down.")


if __name__ == "__main__":
    main()
