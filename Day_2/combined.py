
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

# Prevent Tkinter GUI cleanup errors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
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
    "rpm"
]

TARGET_COLUMN = "state"

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
        "severity": 3
    },
    {
        "name": "HIGH_VIBRATION",
        "priority": 2,
        "condition": lambda r: r["vibration"] > 5.0,
        "action": "CHECK_MACHINE",
        "severity": 2
    },
    {
        "name": "MOTOR_OVERLOAD",
        "priority": 3,
        "condition": lambda r: r["current"] > 18,
        "action": "CHECK_MOTOR",
        "severity": 2
    },
    {
        "name": "HIGH_PRESSURE",
        "priority": 4,
        "condition": lambda r: r["pressure"] > 5.8,
        "action": "REDUCE_PRESSURE",
        "severity": 2
    }
]

ACTION_MESSAGES = {
    "STOP_MACHINE":
        "Stop the machine safely and inspect temperature.",
    "CHECK_MACHINE":
        "Inspect vibration, bearings and alignment.",
    "CHECK_MOTOR":
        "Inspect motor current and possible overload.",
    "REDUCE_PRESSURE":
        "Check pressure limits and control equipment.",
    "CONTINUE_NORMAL":
        "No rule-based fault detected; continue monitoring."
}


# ==========================================================
# STEP 3: LOAD AND CLEAN CSV
# ==========================================================

def load_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"CSV file not found: {DATA_FILE}"
        )

    data = pd.read_csv(
        DATA_FILE,
        encoding="utf-8-sig"
    )

    data.columns = (
        data.columns.astype(str)
        .str.strip()
        .str.lower()
    )

    required = FEATURE_NAMES + [TARGET_COLUMN]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing CSV columns: {missing}"
        )

    X = data[FEATURE_NAMES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    y = (
        data[TARGET_COLUMN]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    valid = (
        np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
        & y.notna().to_numpy()
        & y.ne("").to_numpy()
    )

    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    if X.empty:
        raise ValueError("No valid sensor records found.")

    if y.nunique() < 2:
        raise ValueError(
            "At least two state classes are required for training."
        )

    print("=" * 60)
    print("DAY 2 CSV INFORMATION")
    print("=" * 60)

    print("Valid records:", len(X))
    print("\nState distribution:")
    print(y.value_counts().to_string())

    return X, y


# ==========================================================
# STEP 4: TRAIN RANDOM FOREST
# ==========================================================

def train_model(X, y):

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    counts = y.value_counts()

    can_stratify = (
        counts.min() >= 2
        and int(np.ceil(len(X) * 0.20)) >= y.nunique()
        and int(np.floor(len(X) * 0.80)) >= y.nunique()
    )

    if can_stratify:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        print("\nRANDOM FOREST EVALUATION")
        print("Training samples:", len(X_train))
        print("Testing samples:", len(X_test))
        print(
            "Training accuracy:",
            f"{accuracy_score(y_train, model.predict(X_train)):.2%}"
        )
        print(
            "Test accuracy:",
            f"{accuracy_score(y_test, predictions):.2%}"
        )

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=model.classes_,
                zero_division=0
            )
        )

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=model.classes_
        )

        cm_title = "Random Forest - Test Confusion Matrix"

    else:

        print("\nWARNING:")
        print(
            "Dataset is too small for a reliable stratified "
            "test split. Training on all available records."
        )

        model.fit(X, y)

        predictions = model.predict(X)

        print(
            "Training accuracy (not test accuracy):",
            f"{accuracy_score(y, predictions):.2%}"
        )

        print("\nTraining classification report:")
        print(
            classification_report(
                y,
                predictions,
                labels=model.classes_,
                zero_division=0
            )
        )

        cm = confusion_matrix(
            y,
            predictions,
            labels=model.classes_
        )

        cm_title = "Training Confusion Matrix"

    # Save confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_
    ).plot(
        ax=ax,
        values_format="d",
        xticks_rotation=45,
        colorbar=False
    )

    ax.set_title(cm_title)
    fig.tight_layout()
    fig.savefig(CM_IMAGE, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Save model and feature order
    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "classes": list(model.classes_)
        },
        MODEL_FILE
    )

    # Feature importance
    importance = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    print("\nFeature importance:")
    print(importance.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        importance["Feature"],
        importance["Importance"]
    )

    ax.set_title("Random Forest - Feature Importance")
    ax.set_xlabel("Importance")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(
        IMPORTANCE_IMAGE,
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("\nModel saved:", MODEL_FILE)
    print("Confusion matrix saved:", CM_IMAGE)
    print("Feature importance saved:", IMPORTANCE_IMAGE)

    return model


# ==========================================================
# STEP 5: LOAD EXACTLY FIVE MACHINE READINGS
# ==========================================================

def generate_sensor_data(X):

    if len(X) < NUMBER_OF_MACHINES:
        raise ValueError(
            f"At least {NUMBER_OF_MACHINES} valid CSV records "
            "are required."
        )

    # Use the first five valid records in CSV order.
    machine_data = X.iloc[:NUMBER_OF_MACHINES].reset_index(
        drop=True
    )

    return machine_data


# ==========================================================
# STEP 6: RULE-BASED OUTPUT
# ==========================================================

def evaluate_rules(reading):

    matching_rules = []

    for rule in sorted(
        RULES,
        key=lambda item: item["priority"]
    ):
        if rule["condition"](reading):
            matching_rules.append(rule)

    if not matching_rules:
        return {
            "condition": "NORMAL",
            "action": "CONTINUE_NORMAL",
            "severity": 0,
            "matched_rules": []
        }

    # Highest severity wins; priority breaks ties.
    selected = max(
        matching_rules,
        key=lambda item: (
            item["severity"],
            -item["priority"]
        )
    )

    return {
        "condition": selected["name"],
        "action": selected["action"],
        "severity": selected["severity"],
        "matched_rules": [
            rule["name"] for rule in matching_rules
        ]
    }


# ==========================================================
# STEP 7: MACHINE LEARNING OUTPUT
# ==========================================================

def predict_with_model(model, reading):

    input_data = pd.DataFrame(
        [[reading[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES
    )

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    probability_by_class = {
        str(label): float(probability)
        for label, probability in zip(
            model.classes_,
            probabilities
        )
    }

    confidence = float(max(probabilities))

    if confidence >= 0.80:
        confidence_level = "HIGH"
    elif confidence >= 0.60:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    return {
        "condition": str(prediction),
        "confidence": confidence,
        "confidence_level": confidence_level,
        "probabilities": probability_by_class
    }


# ==========================================================
# STEP 8: PROCESS ONE MACHINE
# ==========================================================

def process_machine(machine, reading, model):

    rule_result = evaluate_rules(reading)

    ml_result = predict_with_model(model, reading)

    # Preserve critical rule alarms.
    if rule_result["severity"] >= 3:
        urgency = "CRITICAL"

        recommendation = (
            "CRITICAL RULE ALARM: Follow approved safety "
            "procedures and verify the machine condition."
        )

    elif rule_result["severity"] == 2:
        urgency = "HIGH"

        recommendation = (
            "Inspect the machine and verify the fault indication."
        )

    elif ml_result["condition"] != rule_result["condition"]:
        urgency = "REVIEW"

        recommendation = (
            "Rule and ML outputs differ. Verify sensor readings."
        )

    else:
        urgency = "NORMAL"

        recommendation = (
            "No fault detected by either method; continue monitoring."
        )

    result = {
        "machine": machine,
        "reading": reading,
        "rule_result": rule_result,
        "ml_result": ml_result,
        "urgency": urgency,
        "recommendation": recommendation
    }

    # Detailed log
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(
            f"\nTimestamp: {datetime.now().isoformat(timespec='seconds')}\n"
            f"Machine: {machine}\n"
            f"Sensor readings: {reading}\n"
            f"Rule output: {rule_result['condition']}\n"
            f"Rule action: {rule_result['action']}\n"
            f"ML output: {ml_result['condition']}\n"
            f"ML confidence: {ml_result['confidence']:.2%}\n"
            f"Urgency: {urgency}\n"
            f"Recommendation: {recommendation}\n"
            + "-" * 60 + "\n"
        )

    return result


# ==========================================================
# STEP 9: MAIN PROGRAM
# ==========================================================

def main():

    # Load the Day 2 CSV
    X, y = load_data()

    # Train Random Forest using CSV sensor values and labels
    model = train_model(X, y)

    # Select exactly five sensor records
    machine_data = generate_sensor_data(X)

    results = []

    print("\n" + "=" * 60)
    print("PROCESSING FIVE MACHINES")
    print("=" * 60)

    for index, row in machine_data.iterrows():

        reading = row.to_dict()

        machine_name = f"Machine {index + 1}"

        print(f"\nProcessing {machine_name}...")

        result = process_machine(
            machine_name,
            reading,
            model
        )

        results.append(result)

    # ======================================================
    # FINAL OUTPUT: RULE-BASED VS MACHINE LEARNING
    # ======================================================

    print("\n\n" + "=" * 65)
    print("FINAL RESULT - FIVE MACHINES")
    print("=" * 65)

    for result in results:

        print(f"\n{result['machine']}")
        print("-" * 40)

        print(
            "Rule-Based Output :",
            result["rule_result"]["condition"]
        )

        print(
            "ML Output         :",
            result["ml_result"]["condition"]
        )

        print(
            "ML Confidence     :",
            f"{result['ml_result']['confidence']:.2%}"
        )

        print(
            "Rule Action       :",
            result["rule_result"]["action"]
        )

        print(
            "Urgency           :",
            result["urgency"]
        )

        print(
            "Recommendation    :",
            result["recommendation"]
        )

    # ======================================================
    # FINAL COMPARISON TABLE
    # ======================================================

    print("\n" + "=" * 65)
    print("FINAL RULE-BASED AND ML COMPARISON")
    print("=" * 65)

    summary = pd.DataFrame([
        {
            "Machine": result["machine"],
            "Rule-Based Output":
                result["rule_result"]["condition"],
            "ML Output":
                result["ml_result"]["condition"],
            "Agreement":
                result["rule_result"]["condition"]
                == result["ml_result"]["condition"],
            "Urgency": result["urgency"]
        }
        for result in results
    ])

    print(summary.to_string(index=False))

    agreements = int(summary["Agreement"].sum())
    disagreements = len(summary) - agreements

    print("\nTotal machines:", len(results))
    print("Agreements:", agreements)
    print("Disagreements:", disagreements)

    # Select the top priority machine.
    priority_order = {
        "CRITICAL": 3,
        "HIGH": 2,
        "REVIEW": 1,
        "NORMAL": 0
    }

    top_priority = max(
        results,
        key=lambda result: (
            priority_order[result["urgency"]],
            result["rule_result"]["severity"],
            result["ml_result"]["confidence"]
        )
    )

    print("\nTOP PRIORITY:", top_priority["machine"])
    print("Rule output:", top_priority["rule_result"]["condition"])
    print("ML output:", top_priority["ml_result"]["condition"])
    print("Urgency:", top_priority["urgency"])
    print("Recommendation:", top_priority["recommendation"])

    print("\nDetailed log:", LOG_FILE)
    print("Model file:", MODEL_FILE)
    print("Confusion matrix:", CM_IMAGE)
    print("Feature importance:", IMPORTANCE_IMAGE)

    print("=" * 65)


# ==========================================================
# PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
