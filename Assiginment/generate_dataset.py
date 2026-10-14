"""
Robotic Arm Joint Health Monitoring System
-------------------------------------------
Bonus Challenge - synthetic historical dataset generator.

The company gave us nothing, so we build our own history of past joint
failures. Each row is ONE joint reading (torque, temperature, vibration)
taken from a robotic-arm joint at some point in its life, together with
the wear label that a technician assigned after inspection.

The three wear classes are engineered to OVERLAP slightly and to rise
together (a worn joint runs hotter, draws more torque and vibrates more),
because that correlation is exactly what the spoof defence later relies on.

Run once to (re)create  data/joint_failure_history.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = DATA_DIR / "joint_failure_history.csv"

# A fixed seed keeps the committed CSV reproducible.
RANDOM_SEED = 24461

JOINTS = ["shoulder", "elbow", "wrist"]

# Every class stays BELOW the hard safety limits used by the
# immediate-stop check (torque 130 Nm, temp 85 C, vibration 8 mm/s).
# That is the whole point of "predict wear BEFORE it becomes a safety
# issue": the model learns the danger zone that sits just under the line.
#
# class : (count, torque range, temperature range, vibration range)
CLASS_PROFILES = {
    "HEALTHY": {
        "count": 24,
        "torque": (20.0, 70.0),
        "temperature": (32.0, 56.0),
        "vibration": (0.5, 3.0),
    },
    "WEAR": {
        "count": 20,
        "torque": (60.0, 95.0),
        "temperature": (55.0, 70.0),
        "vibration": (3.0, 5.8),
    },
    "CRITICAL": {
        "count": 16,
        "torque": (90.0, 125.0),
        "temperature": (70.0, 84.0),
        "vibration": (5.8, 7.9),
    },
}


# ==========================================================
# GENERATION
# ==========================================================

def generate_rows(rng):
    """Build one list of reading dicts across all three wear classes."""
    rows = []

    for wear_label, profile in CLASS_PROFILES.items():
        for _ in range(profile["count"]):
            rows.append({
                "joint": rng.choice(JOINTS),
                "torque": round(rng.uniform(*profile["torque"]), 2),
                "temperature": round(rng.uniform(*profile["temperature"]), 2),
                "vibration": round(rng.uniform(*profile["vibration"]), 2),
                "wear_label": wear_label,
            })

    # Shuffle so the classes are not stored in blocks.
    rng.shuffle(rows)
    return rows


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)
    rows = generate_rows(rng)

    frame = pd.DataFrame(
        rows,
        columns=["joint", "torque", "temperature", "vibration", "wear_label"],
    )

    frame.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("=" * 60)
    print("SYNTHETIC JOINT-FAILURE HISTORY GENERATED")
    print("=" * 60)
    print("Rows written :", len(frame))
    print("Output file  :", OUTPUT_FILE)
    print("\nClass distribution:")
    print(frame["wear_label"].value_counts().to_string())
    print("\nFirst five rows:")
    print(frame.head().to_string(index=False))


if __name__ == "__main__":
    main()
