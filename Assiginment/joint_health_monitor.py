"""
Robotic Arm Joint Health Monitoring System
===========================================
23IF02 System Automation - Bonus Challenge.

A fleet of 6-axis robotic arms is monitored on an assembly line. Each arm
exposes three joints (shoulder, elbow, wrist); each joint reports torque,
temperature and vibration. The system fuses FOUR independent signals into
one decision per arm:

    1. Immediate safety stop   - deterministic thresholds, NEVER uses the ML
                                 model. Works even if the model is missing.
    2. Predictive wear         - RandomForest trained on historical joint
                                 failures (data/joint_failure_history.csv).
    3. Maintenance notes       - keyword scan of a free-text technician log.
    4. Spoofed-sensor defence  - physics consistency check that catches a
                                 faked low torque reading used to hide load.

Final decision per arm is one of:
    SAFE | MONITOR | SCHEDULE_MAINTENANCE | EMERGENCY_STOP

Run:  python joint_health_monitor.py
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================================
# STEP 1: CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

HISTORY_FILE = DATA_DIR / "joint_failure_history.csv"
FLEET_FILE = DATA_DIR / "current_fleet_readings.csv"
LOG_FILE = DATA_DIR / "maintenance_log.txt"

MODEL_FILE = BASE_DIR / "joint_wear_model.pkl"
REPORT_FILE = BASE_DIR / "fleet_decision_report.txt"

FEATURE_NAMES = ["torque", "temperature", "vibration"]
TARGET_COLUMN = "wear_label"

JOINTS = ["shoulder", "elbow", "wrist"]


# --- Requirement 1: HARD safety limits (deterministic, model-free) --------
# A joint crossing ANY of these is an instant physical-danger stop.
# Justification lives in EXPLANATION.md. These sit ABOVE the worst
# "CRITICAL wear" readings in the training data on purpose, so that the
# model can warn us in the gap between "worn" and "dangerous".
SAFETY_LIMITS = {
    "torque": 130.0,       # Nm  - mechanical overload of the actuator/gearing
    "temperature": 85.0,   # C   - grease breakdown / servo insulation risk
    "vibration": 8.0,      # mm/s RMS - ISO 10816 "unacceptable" severity zone
}


# --- Requirement 4: spoof-detection band ----------------------------------
# A genuinely stressed joint is hot AND vibrating AND pulling high torque,
# because heat and vibration are physical SIDE EFFECTS of real load. If a
# joint is clearly hot and shaking but its torque reads low/normal, the
# torque channel is physically implausible -> treat it as spoofed.
SPOOF_TEMP_HIGH = 68.0      # C    - joint is clearly running hot
SPOOF_VIB_HIGH = 5.5        # mm/s - joint is clearly shaking
SPOOF_TORQUE_LOW = 60.0     # Nm   - but torque claims a light, easy load
# Basic physical plausibility bounds (anything outside is impossible/garbage).
PLAUSIBLE_TORQUE = (0.0, 200.0)


# --- Requirement 3: maintenance-note keywords -----------------------------
# Words a technician uses when something is MECHANICALLY wrong with a joint.
# Housekeeping notes (lights, coffee, paint, fire drill) contain none of
# these and are therefore ignored automatically.
CONCERN_KEYWORDS = [
    "grind", "grinding", "noise", "noisy", "loud", "squeak", "squeal",
    "rattle", "knock", "leak", "leaking", "oil", "grease", "loose", "play",
    "backlash", "wobble", "overheat", "hot", "warm", "burning", "smell",
    "crack", "vibrat", "stiff", "jerky", "slip", "flat", "steady",
]


# ==========================================================
# STEP 2: TRAIN / LOAD THE PREDICTIVE WEAR MODEL
#         (Requirement 2)
# ==========================================================

def train_wear_model():
    """Train a RandomForest on the historical joint-failure dataset."""
    if not HISTORY_FILE.exists():
        raise FileNotFoundError(
            f"History not found: {HISTORY_FILE}\n"
            "Run generate_dataset.py first."
        )

    data = pd.read_csv(HISTORY_FILE, encoding="utf-8-sig")
    data.columns = data.columns.astype(str).str.strip().str.lower()

    X = data[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    y = data[TARGET_COLUMN].astype("string").str.strip().str.upper()

    valid = (
        pd.Series(np.isfinite(X.to_numpy(dtype=float)).all(axis=1), index=X.index)
        & y.notna()
        & y.ne("")
    )
    X, y = X.loc[valid].copy(), y.loc[valid].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print("=" * 64)
    print("PREDICTIVE WEAR MODEL  (Requirement 2)")
    print("=" * 64)
    print("Historical readings :", len(X))
    print("Training / testing   :", len(X_train), "/", len(X_test))
    print("Classes              :", ", ".join(model.classes_))
    print(f"Test accuracy        : {accuracy_score(y_test, model.predict(X_test)):.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, model.predict(X_test), zero_division=0))

    joblib.dump(
        {"model": model, "feature_names": FEATURE_NAMES,
         "classes": list(model.classes_)},
        MODEL_FILE,
    )
    print("Model saved to       :", MODEL_FILE)

    return model


def predict_joint_wear(model, torque, temperature, vibration):
    """Return (wear_label, confidence) for a single joint reading."""
    row = pd.DataFrame([[torque, temperature, vibration]], columns=FEATURE_NAMES)
    label = str(model.predict(row)[0])
    confidence = float(np.max(model.predict_proba(row)[0]))
    return label, confidence


# ==========================================================
# STEP 3: IMMEDIATE SAFETY STOP  (Requirement 1)
#         Pure thresholds. No model. Cannot be skipped.
# ==========================================================

def check_safety(arm_joints):
    """
    arm_joints: dict {joint_name: {torque, temperature, vibration}}
    Returns (is_danger, list_of_reasons). Deterministic and model-free.
    """
    reasons = []
    for joint, reading in arm_joints.items():
        for sensor, limit in SAFETY_LIMITS.items():
            value = reading[sensor]
            if value > limit:
                reasons.append(
                    f"{joint} {sensor}={value:g} exceeds hard limit {limit:g}"
                )
    return (len(reasons) > 0), reasons


# ==========================================================
# STEP 4: SPOOFED-SENSOR DEFENCE  (Requirement 4)
# ==========================================================

def check_spoofing(arm_joints):
    """
    Catch a faked torque reading that hides a real overload.

    A real, heavily loaded joint heats up and vibrates. If temperature and
    vibration both say "high stress" while torque reads low, the torque
    channel contradicts the physics -> flag it. Also reject torque values
    outside the physically possible range.

    Returns (is_spoof, list_of_reasons).
    """
    reasons = []
    for joint, reading in arm_joints.items():
        torque = reading["torque"]
        temperature = reading["temperature"]
        vibration = reading["vibration"]

        low, high = PLAUSIBLE_TORQUE
        if not (low <= torque <= high):
            reasons.append(
                f"{joint} torque={torque:g} is outside the physically "
                f"possible range {low:g}-{high:g}"
            )
            continue

        if (
            temperature >= SPOOF_TEMP_HIGH
            and vibration >= SPOOF_VIB_HIGH
            and torque < SPOOF_TORQUE_LOW
        ):
            reasons.append(
                f"{joint} is hot ({temperature:g} C) and vibrating "
                f"({vibration:g} mm/s) yet torque reads only {torque:g} Nm "
                f"- torque channel is physically inconsistent (possible spoof)"
            )
    return (len(reasons) > 0), reasons


# ==========================================================
# STEP 5: MAINTENANCE NOTES AS A SECOND SIGNAL  (Requirement 3)
# ==========================================================

def scan_maintenance_log():
    """
    Parse the free-text log into a per-arm concern signal.

    Returns {arm_id: [matched note strings]}. A note counts only if it both
    mentions an arm ("Arm 3") AND contains a mechanical concern keyword, so
    housekeeping lines are dropped automatically.
    """
    concerns = {}
    if not LOG_FILE.exists():
        print("WARNING: maintenance log not found, skipping note signal.")
        return concerns

    for raw_line in LOG_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        lowered = line.lower()

        # Which arm (if any) does this note talk about?
        arm_id = None
        for number in range(1, 7):
            if f"arm {number}" in lowered or f"arm_{number}" in lowered:
                arm_id = f"Arm_{number}"
                break
        if arm_id is None:
            continue  # housekeeping note with no arm -> ignore

        matched = [kw for kw in CONCERN_KEYWORDS if kw in lowered]
        if matched:
            concerns.setdefault(arm_id, []).append(line)

    return concerns


# ==========================================================
# STEP 6: COMBINE EVERYTHING INTO ONE DECISION PER ARM
#         (Requirement 5)
# ==========================================================

def decide_for_arm(arm_id, arm_joints, model, note_concerns):
    """Fuse the four signals into SAFE / MONITOR / SCHEDULE_MAINTENANCE /
    EMERGENCY_STOP, in strict priority order."""

    # --- Signal 1: deterministic safety (always first, no model) ----------
    is_danger, safety_reasons = check_safety(arm_joints)

    # --- Signal 4: sensor integrity --------------------------------------
    is_spoof, spoof_reasons = check_spoofing(arm_joints)

    # --- Signal 2: predictive wear (worst joint on the arm) --------------
    worst_label, worst_conf, worst_joint = "HEALTHY", 0.0, None
    rank = {"HEALTHY": 0, "WEAR": 1, "CRITICAL": 2}
    ml_per_joint = {}
    for joint, reading in arm_joints.items():
        label, conf = predict_joint_wear(
            model, reading["torque"], reading["temperature"], reading["vibration"]
        )
        ml_per_joint[joint] = (label, conf)
        if rank[label] > rank[worst_label]:
            worst_label, worst_conf, worst_joint = label, conf, joint

    # --- Signal 3: maintenance notes -------------------------------------
    notes = note_concerns.get(arm_id, [])
    has_note_concern = len(notes) > 0

    # --- Decision tree ----------------------------------------------------
    if is_danger:
        decision = "EMERGENCY_STOP"
        rationale = "Hard safety limit crossed: " + "; ".join(safety_reasons)

    elif is_spoof:
        # Torque is untrustworthy. Fall back to the channels we CAN trust
        # (temperature, vibration). If those alone are already dangerous,
        # stop now; otherwise require a physical inspection.
        trusted_danger = any(
            r["temperature"] > SAFETY_LIMITS["temperature"]
            or r["vibration"] > SAFETY_LIMITS["vibration"]
            for r in arm_joints.values()
        )
        if trusted_danger:
            decision = "EMERGENCY_STOP"
            rationale = ("Sensor spoof suspected AND trustworthy channels "
                         "already dangerous: " + "; ".join(spoof_reasons))
        else:
            decision = "SCHEDULE_MAINTENANCE"
            rationale = ("Sensor spoof suspected - torque cannot be trusted, "
                         "physical inspection required: " + "; ".join(spoof_reasons))

    elif worst_label == "CRITICAL" and worst_conf >= 0.60:
        decision = "SCHEDULE_MAINTENANCE"
        rationale = (f"Model predicts CRITICAL wear at {worst_joint} "
                     f"(confidence {worst_conf:.0%})")
        if has_note_concern:
            rationale += " and technician notes corroborate"

    elif worst_label == "WEAR" and has_note_concern:
        decision = "SCHEDULE_MAINTENANCE"
        rationale = (f"Model predicts developing wear at {worst_joint} "
                     f"AND technician notes flag a concern")

    elif worst_label in ("WEAR", "CRITICAL") or has_note_concern:
        decision = "MONITOR"
        bits = []
        if worst_label in ("WEAR", "CRITICAL"):
            bits.append(f"model sees {worst_label.lower()} at {worst_joint} "
                        f"(confidence {worst_conf:.0%})")
        if has_note_concern:
            bits.append("technician note flags a concern")
        rationale = "Early warning: " + "; ".join(bits)

    else:
        decision = "SAFE"
        rationale = "All signals nominal; no model, note or physics warning"

    return {
        "arm_id": arm_id,
        "decision": decision,
        "rationale": rationale,
        "safety_reasons": safety_reasons,
        "spoof_reasons": spoof_reasons,
        "ml_per_joint": ml_per_joint,
        "worst": (worst_joint, worst_label, worst_conf),
        "notes": notes,
    }


# ==========================================================
# STEP 7: LOAD CURRENT FLEET READINGS
# ==========================================================

def load_fleet():
    """Return {arm_id: {joint: {torque, temperature, vibration}}}."""
    if not FLEET_FILE.exists():
        raise FileNotFoundError(f"Fleet readings not found: {FLEET_FILE}")

    data = pd.read_csv(FLEET_FILE, encoding="utf-8-sig")
    data.columns = data.columns.astype(str).str.strip().str.lower()

    fleet = {}
    for _, row in data.iterrows():
        arm_id = str(row["arm_id"]).strip()
        joint = str(row["joint"]).strip().lower()
        fleet.setdefault(arm_id, {})[joint] = {
            "torque": float(row["torque"]),
            "temperature": float(row["temperature"]),
            "vibration": float(row["vibration"]),
        }
    return fleet


# ==========================================================
# STEP 8: MAIN PROGRAM
# ==========================================================

DECISION_ICON = {
    "SAFE": "[ SAFE ]",
    "MONITOR": "[ MONITOR ]",
    "SCHEDULE_MAINTENANCE": "[ SCHEDULE_MAINTENANCE ]",
    "EMERGENCY_STOP": "[ ** EMERGENCY_STOP ** ]",
}


def main():
    model = train_wear_model()

    fleet = load_fleet()
    note_concerns = scan_maintenance_log()

    print("\n" + "=" * 64)
    print("MAINTENANCE NOTE SIGNAL  (Requirement 3)")
    print("=" * 64)
    if note_concerns:
        for arm_id, notes in sorted(note_concerns.items()):
            print(f"{arm_id}: {len(notes)} relevant note(s)")
    else:
        print("No arm-specific mechanical concerns found in the log.")

    results = []
    for arm_id in sorted(fleet):
        results.append(
            decide_for_arm(arm_id, fleet[arm_id], model, note_concerns)
        )

    # ---- Per-arm report --------------------------------------------------
    print("\n" + "=" * 64)
    print("PER-ARM DECISIONS  (Requirement 5)")
    print("=" * 64)

    report_lines = [
        f"Robotic Arm Fleet Decision Report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "=" * 64,
    ]

    for res in results:
        joint_txt = ", ".join(
            f"{j}:{lab}({conf:.0%})" for j, (lab, conf) in res["ml_per_joint"].items()
        )
        block = [
            f"\n{res['arm_id']}  ->  {DECISION_ICON[res['decision']]}",
            f"   Why        : {res['rationale']}",
            f"   Wear (ML)  : {joint_txt}",
        ]
        if res["notes"]:
            block.append(f"   Tech notes : {res['notes'][0]}")
        for line in block:
            print(line)
            report_lines.append(line)

    # ---- Fleet summary ---------------------------------------------------
    print("\n" + "=" * 64)
    print("FLEET SUMMARY")
    print("=" * 64)

    summary = pd.DataFrame([
        {"Arm": r["arm_id"], "Decision": r["decision"]} for r in results
    ])
    print(summary.to_string(index=False))

    counts = summary["Decision"].value_counts()
    print("\nDecision counts:")
    for decision in ["EMERGENCY_STOP", "SCHEDULE_MAINTENANCE", "MONITOR", "SAFE"]:
        print(f"  {decision:<22}: {int(counts.get(decision, 0))}")

    reached = set(summary["Decision"])
    all_four = {"SAFE", "MONITOR", "SCHEDULE_MAINTENANCE", "EMERGENCY_STOP"}
    print("\nAll four outcomes reached in this run:",
          "YES" if all_four <= reached else f"NO (missing {all_four - reached})")

    report_lines.append("\n" + summary.to_string(index=False))
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")
    print("\nFull report saved to:", REPORT_FILE)


if __name__ == "__main__":
    main()
