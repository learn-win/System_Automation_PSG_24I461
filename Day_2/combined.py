
from pathlib import Path
from datetime import datetime
import random
import time

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ==========================================================
# STEP 1: CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "day2_sample.csv"

MODEL_FILE = BASE_DIR / "random_forest_model.pkl"
LOG_FILE = BASE_DIR / "combined_machine_log.txt"
CM_IMAGE = BASE_DIR / "random_forest_confusion_matrix.png"
IMPORTANCE_IMAGE = BASE_DIR / "feature_importance.png"

FEATURE_NAMES = [
    "temperature",
    "pressure",
    "vibration",
    "current",
    "rpm",
]

TARGET_COLUMN = "state"

# Set to True to generate five simulated machines.
# Set to False to enter sensor values manually.
SIMULATE_MACHINES = True

NUMBER_OF_MACHINES = 5


# ==========================================================
# STEP 2: RULE DEFINITIONS
# ==========================================================

RULES = [
    {
        "name": "OVERHEATING",
        "priority": 1,
        "condition": lambda r: r["temperature"] > 80,
        "action": "STOP_MACHINE",
        "severity": 3,
    },
    {
        "name": "HIGH_VIBRATION",
        "priority": 2,
        "condition": lambda r: r["vibration"] > 5.0,
        "action": "CHECK_MACHINE",
        "severity": 2,
    },
    {
        "name": "MOTOR_OVERLOAD",
        "priority": 3,
        "condition": lambda r: r["current"] > 18,
        "action": "CHECK_MOTOR",
        "severity": 2,
    },
    {
        "name": "HIGH_PRESSURE",
        "priority": 4,
        "condition": lambda r: r["pressure"] > 5.8,
        "action": "REDUCE_PRESSURE",
        "severity": 2,
    },
]

ACTION_MESSAGES = {
    "STOP_MACHINE":
        "Stop the machine safely and inspect temperature.",
    "CHECK_MACHINE":
        "Inspect vibration, mounting, bearings and alignment.",
    "CHECK_MOTOR":
        "Inspect motor current and possible overload.",
    "REDUCE_PRESSURE":
        "Check pressure limits and pressure-control equipment.",
    "CONTINUE_NORMAL":
        "No rule-based fault detected; continue monitoring.",
}


# ==========================================================
# STEP 3: LOAD AND CLEAN DATA
# ==========================================================

def load_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    data = pd.read_csv(
        DATA_FILE,
        encoding="utf-8-sig",
    )

    data.columns = (
        data.columns.astype(str)
        .str.strip()
        .str.lower()
    )

    expected = FEATURE_NAMES + [TARGET_COLUMN]

    missing = [
        column for column in expected
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing CSV columns: {missing}\n"
            f"Available columns: {data.columns.tolist()}"
        )

    X = data[FEATURE_NAMES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    y = (
        data[TARGET_COLUMN]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    valid = (
        pd.Series(
            np.isfinite(X.to_numpy(dtype=float)).all(axis=1),
            index=X.index,
        )
        & y.notna()
        & y.ne("")
    )

    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    print("=" * 65)
    print("DATASET INFORMATION")
    print("=" * 65)

    print("Total rows:", len(data))
    print("Valid rows:", len(X))
    print("Invalid rows:", int((~valid).sum()))

    print("\nMachine condition distribution:")
    print(y.value_counts().to_string())

    if X.empty:
        raise ValueError("No valid dataset rows found.")

    if y.nunique() < 2:
        raise ValueError(
            "At least two machine condition classes are required."
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            "Every class needs at least two samples. "
            "Collect more examples for rare conditions."
        )

    return X, y


# ==========================================================
# STEP 4: TRAIN RANDOM FOREST
# ==========================================================

def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(
        y_train, model.predict(X_train)
    )

    test_predictions = model.predict(X_test)

    test_accuracy = accuracy_score(
        y_test, test_predictions
    )

    print("\n" + "=" * 65)
    print("RANDOM FOREST TRAINING")
    print("=" * 65)

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Number of trees:", model.n_estimators)
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=model.classes_,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_test,
        test_predictions,
        labels=model.classes_,
    )

    print("Confusion matrix:")
    print(
        pd.DataFrame(
            cm,
            index=model.classes_,
            columns=model.classes_,
        )
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_,
    )

    display.plot(values_format="d", xticks_rotation=45)
    plt.title("Random Forest - Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CM_IMAGE, dpi=180, bbox_inches="tight")
    plt.close()

    print("Confusion matrix saved:", CM_IMAGE)

    # Save the trained model and feature order.
    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "classes": list(model.classes_),
        },
        MODEL_FILE,
    )

    print("Model saved:", MODEL_FILE)

    # Show feature importance.
    importance = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    print("\nSensor feature importance:")
    print(importance.to_string(index=False))

    plt.figure(figsize=(9, 5))
    plt.barh(
        importance["Feature"],
        importance["Importance"],
    )
    plt.xlabel("Importance")
    plt.title("Random Forest - Sensor Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(
        IMPORTANCE_IMAGE,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    return model


# ==========================================================
# STEP 5: GENERATE SENSOR READINGS
# ==========================================================

def generate_sensor_data():

    # Example ranges are illustrative and should be calibrated
    # to your actual machine and sensor operating limits.
    return {
        "temperature": round(random.uniform(50, 95), 2),
        "pressure": round(random.uniform(4.0, 6.2), 2),
        "vibration": round(random.uniform(0.1, 7.0), 2),
        "current": round(random.uniform(5, 25), 2),
        "rpm": random.randint(1400, 2200),
    }


# ==========================================================
# STEP 6: RULE-BASED REASONING
# ==========================================================

def evaluate_rules(reading):

    matching_rules = []

    for rule in sorted(
        RULES,
        key=lambda item: item["priority"],
    ):
        if rule["condition"](reading):
            matching_rules.append(rule)

    if not matching_rules:
        return {
            "condition": "NORMAL",
            "rule": "NO_RULE_FIRED",
            "action": "CONTINUE_NORMAL",
            "severity": 0,
            "priority": 999,
            "matched_rules": [],
        }

    # Highest severity wins. Priority breaks severity ties.
    worst_rule = max(
        matching_rules,
        key=lambda item: (
            item["severity"],
            -item["priority"],
        ),
    )

    return {
        "condition": worst_rule["name"],
        "rule": worst_rule["name"],
        "action": worst_rule["action"],
        "severity": worst_rule["severity"],
        "priority": worst_rule["priority"],
        "matched_rules": [
            rule["name"] for rule in matching_rules
        ],
    }


# ==========================================================
# STEP 7: MACHINE LEARNING PREDICTION
# ==========================================================

def predict_with_model(model, reading):

    input_data = pd.DataFrame(
        [[reading[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    probability_by_class = {
        str(label): float(probability)
        for label, probability in zip(
            model.classes_,
            probabilities,
        )
    }

    confidence = max(probabilities)

    # Confidence bands are operational thresholds, not guarantees.
    if confidence >= 0.80:
        confidence_level = "HIGH"
    elif confidence >= 0.60:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    return {
        "condition": str(prediction),
        "confidence": float(confidence),
        "confidence_level": confidence_level,
        "probabilities": probability_by_class,
    }


# ==========================================================
# STEP 8: COMPARE RULE ENGINE WITH RANDOM FOREST
# ==========================================================

def compare_decisions(rule_result, model_result):

    rule_condition = rule_result["condition"]
    model_condition = model_result["condition"]

    agrees = rule_condition == model_condition

    if agrees:
        comparison = "AGREEMENT"
    else:
        comparison = "DISAGREEMENT"

    # Convert model prediction to an operational severity.
    model_severity_map = {
        "NORMAL": 0,
        "OVERHEATING": 3,
        "HIGH_VIBRATION": 2,
        "MOTOR_OVERLOAD": 2,
        "HIGH_PRESSURE": 2,
    }

    model_severity = model_severity_map.get(
        model_condition,
        0,
    )

    # Combined urgency is the maximum severity from either system.
    combined_severity = max(
        rule_result["severity"],
        model_severity,
    )

    if combined_severity >= 3:
        urgency = "CRITICAL"
    elif combined_severity == 2:
        urgency = "HIGH"
    elif combined_severity == 1:
        urgency = "MEDIUM"
    else:
        urgency = "NORMAL"

    # Surface disagreement for human review.
    if not agrees:
        recommendation = (
            "REVIEW REQUIRED: Rule and ML predictions differ. "
            "Verify sensor readings and inspect the machine."
        )
    elif combined_severity >= 3:
        recommendation = (
            "Urgent fault indication. Follow approved safety "
            "procedures and verify before intervention."
        )
    elif combined_severity > 0:
        recommendation = (
            "Inspect the machine and verify the fault indication."
        )
    else:
        recommendation = (
            "No fault detected by either method; continue monitoring."
        )

    return {
        "comparison": comparison,
        "agrees": agrees,
        "model_severity": model_severity,
        "combined_severity": combined_severity,
        "urgency": urgency,
        "recommendation": recommendation,
    }


# ==========================================================
# STEP 9: LOG EACH MACHINE RESULT
# ==========================================================

def log_decision(
    machine,
    reading,
    rule_result,
    model_result,
    comparison_result,
):

    with open(LOG_FILE, "a", encoding="utf-8") as file:

        file.write(
            f"\nTimestamp: {datetime.now().isoformat(timespec='seconds')}\n"
            f"Machine: {machine}\n"
            f"Sensor readings: {reading}\n"
            f"Rule condition: {rule_result['condition']}\n"
            f"Matched rules: {rule_result['matched_rules']}\n"
            f"Rule action: {rule_result['action']}\n"
            f"Rule severity: {rule_result['severity']}\n"
            f"ML prediction: {model_result['condition']}\n"
            f"ML confidence: {model_result['confidence']:.2%}\n"
            f"ML confidence level: {model_result['confidence_level']}\n"
            f"ML class probabilities: {model_result['probabilities']}\n"
            f"Comparison: {comparison_result['comparison']}\n"
            f"Urgency: {comparison_result['urgency']}\n"
            f"Recommendation: {comparison_result['recommendation']}\n"
            + "-" * 65 + "\n"
        )


# ==========================================================
# STEP 10: PROCESS ONE MACHINE
# ==========================================================

def process_machine(machine, reading, model):

    rule_result = evaluate_rules(reading)
    model_result = predict_with_model(model, reading)

    comparison_result = compare_decisions(
        rule_result,
        model_result,
    )

    print("\n" + "=" * 65)
    print(machine)
    print("=" * 65)

    print("Sensor readings:")
    for name, value in reading.items():
        print(f"  {name}: {value}")

    print("\nRULE ENGINE")
    print("Rule condition :", rule_result["condition"])
    print("Matched rules  :", rule_result["matched_rules"])
    print("Required action:", rule_result["action"])
    print("Rule severity  :", rule_result["severity"])

    print("\nRANDOM FOREST")
    print("Predicted state:", model_result["condition"])
    print(
        f"Prediction confidence: "
        f"{model_result['confidence'] * 100:.2f}%"
    )
    print("Confidence level:", model_result["confidence_level"])

    print("\nClass probabilities:")
    for label, probability in model_result["probabilities"].items():
        print(f"  {label}: {probability * 100:.2f}%")

    print("\nRULE vs MODEL")
    print("Comparison:", comparison_result["comparison"])
    print("Urgency:", comparison_result["urgency"])
    print("Recommendation:", comparison_result["recommendation"])

    log_decision(
        machine,
        reading,
        rule_result,
        model_result,
        comparison_result,
    )

    return {
        "machine": machine,
        "reading": reading,
        "rule_result": rule_result,
        "model_result": model_result,
        "comparison_result": comparison_result,
    }


# ==========================================================
# STEP 11: MANUAL SENSOR INPUT
# ==========================================================

def get_manual_reading():

    reading = {}

    for feature in FEATURE_NAMES:
        reading[feature] = float(
            input(f"Enter {feature}: ")
        )

    if not np.isfinite(
        [reading[name] for name in FEATURE_NAMES]
    ).all():
        raise ValueError("Sensor readings must be finite numbers.")

    return reading


# ==========================================================
# STEP 12: MAIN PROGRAM
# ==========================================================

def main():

    # Train the ML model first.
    X, y = load_data()
    model = train_model(X, y)

    results = []

    print("\n" + "=" * 65)
    print("RULE ENGINE + RANDOM FOREST PREDICTIVE MAINTENANCE")
    print("=" * 65)

    if SIMULATE_MACHINES:

        machines = [
            f"Machine {i}"
            for i in range(1, NUMBER_OF_MACHINES + 1)
        ]

        for machine in machines:
            reading = generate_sensor_data()

            result = process_machine(
                machine,
                reading,
                model,
            )

            results.append(result)
            time.sleep(0.2)

    else:

        try:
            reading = get_manual_reading()

            result = process_machine(
                "Manual Machine",
                reading,
                model,
            )

            results.append(result)

        except ValueError as error:
            print("Invalid input:", error)
            return

    # ------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL MACHINE PRIORITY SUMMARY")
    print("=" * 65)

    # Sort by combined severity, then model confidence.
    ordered_results = sorted(
        results,
        key=lambda item: (
            item["comparison_result"]["combined_severity"],
            item["model_result"]["confidence"],
        ),
        reverse=True,
    )

    for item in ordered_results:

        machine = item["machine"]
        rule_condition = item["rule_result"]["condition"]
        model_condition = item["model_result"]["condition"]
        comparison = item["comparison_result"]["comparison"]
        urgency = item["comparison_result"]["urgency"]
        confidence = item["model_result"]["confidence"]

        print(
            f"{machine} | "
            f"Rule={rule_condition} | "
            f"ML={model_condition} | "
            f"Result={comparison} | "
            f"Urgency={urgency} | "
            f"ML Confidence={confidence:.1%}"
        )

    # Identify the machine requiring the most urgent review.
    most_urgent = ordered_results[0]

    print("\nTOP PRIORITY MACHINE:", most_urgent["machine"])
    print(
        "Recommended response:",
        most_urgent["comparison_result"]["recommendation"],
    )

    disagreements = [
        item for item in results
        if not item["comparison_result"]["agrees"]
    ]

    print("\nDecision agreements:", len(results) - len(disagreements))
    print("Decision disagreements:", len(disagreements))

    if disagreements:
        print("\nMachines requiring decision review:")
        for item in disagreements:
            print(
                f"- {item['machine']}: "
                f"Rule={item['rule_result']['condition']}, "
                f"ML={item['model_result']['condition']}"
            )

    print("\nDetailed log saved to:", LOG_FILE)
    print("Trained model saved to:", MODEL_FILE)
    print("=" * 65)


if __name__ == "__main__":
    main()
