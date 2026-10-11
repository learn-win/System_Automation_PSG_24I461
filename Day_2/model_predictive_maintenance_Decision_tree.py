
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ==========================================================
# STEP 1: PROJECT CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "day2_sample.csv"

TREE_IMAGE = BASE_DIR / "visual_decision_tree_rules.png"
MODEL_FILE = BASE_DIR / "trained_model_decision_tree.pkl"
RULES_FILE = BASE_DIR / "decision_tree_rules.txt"

FEATURE_NAMES = [
    "temperature",
    "pressure",
    "vibration",
    "current",
    "rpm",
]

TARGET_COLUMN = "state"

# ==========================================================
# STEP 2: LOAD AND CLEAN DATASET
# ==========================================================

def load_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    # Your CSV already contains a header.
    data = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

    # Normalize column names.
    data.columns = (
        data.columns.astype(str)
        .str.strip()
        .str.lower()
    )

    expected_columns = FEATURE_NAMES + [TARGET_COLUMN]

    missing = [
        column for column in expected_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available columns: {data.columns.tolist()}"
        )

    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print("Total rows:", len(data))
    print("\nFirst five rows:")
    print(data[expected_columns].head().to_string(index=False))

    # Convert sensor features into numeric values.
    X = data[FEATURE_NAMES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Clean target labels.
    y = (
        data[TARGET_COLUMN]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # Remove missing/infinite sensor readings and empty labels.
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

    print("\nValid rows:", len(X))
    print("Invalid rows:", int((~valid).sum()))

    print("\nMachine condition counts:")
    print(y.value_counts().to_string())

    if X.empty:
        raise ValueError(
            "No valid rows remain. Check the CSV data."
        )

    if y.nunique() < 2:
        raise ValueError(
            "At least two different machine conditions "
            "are needed to train the classifier."
        )

    # Ensure every class has enough samples for a stratified split.
    if y.value_counts().min() < 2:
        raise ValueError(
            "Each machine condition needs at least two samples "
            "for a stratified train/test split. Collect more data."
        )

    return X, y


# ==========================================================
# STEP 3: TRAIN THE DECISION TREE
# ==========================================================

def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_leaf=2,
        random_state=42,
    )

    model.fit(X_train, y_train)

    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Tree depth:", model.get_depth())
    print("Number of leaf nodes:", model.get_n_leaves())

    return model, X_test, y_test


# ==========================================================
# STEP 4: EVALUATE THE MODEL
# ==========================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=model.classes_,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_,
    )

    print("\nConfusion Matrix:")
    print(pd.DataFrame(
        cm,
        index=model.classes_,
        columns=model.classes_,
    ))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_,
    )

    display.plot(values_format="d", xticks_rotation=45)
    plt.title("Machine Condition - Confusion Matrix")
    plt.tight_layout()
    plt.show()
    plt.close()


# ==========================================================
# STEP 5: DISPLAY AND SAVE DECISION TREE RULES
# ==========================================================

def display_rules(model):

    print("\n" + "=" * 60)
    print("DECISION TREE RULES")
    print("=" * 60)

    rules = export_text(
        model,
        feature_names=FEATURE_NAMES,
        decimals=3,
    )

    print(rules)

    RULES_FILE.write_text(
        rules,
        encoding="utf-8",
    )

    print("\nRules saved to:", RULES_FILE)


# ==========================================================
# STEP 6: VISUALIZE THE DECISION TREE
# ==========================================================

def save_tree(model):

    plt.figure(figsize=(26, 15))

    plot_tree(
        model,
        feature_names=FEATURE_NAMES,
        class_names=[str(c) for c in model.classes_],
        filled=True,
        rounded=True,
        impurity=True,
        precision=2,
        fontsize=9,
    )

    plt.title(
        "Decision Tree - Predictive Machine Maintenance",
        fontsize=16,
    )

    plt.tight_layout()

    plt.savefig(
        TREE_IMAGE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    print("\nTree image saved to:", TREE_IMAGE)


# ==========================================================
# STEP 7: SAVE THE MODEL
# ==========================================================

def save_model(model):

    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "class_names": list(model.classes_),
    }

    joblib.dump(artifact, MODEL_FILE)

    print("Trained model saved to:", MODEL_FILE)


# ==========================================================
# STEP 8: PREDICT A NEW MACHINE CONDITION
# ==========================================================

def predict_machine(model):

    print("\n" + "=" * 60)
    print("PREDICT MACHINE CONDITION")
    print("=" * 60)

    try:
        temperature = float(input("Enter temperature: "))
        pressure = float(input("Enter pressure: "))
        vibration = float(input("Enter vibration: "))
        current = float(input("Enter current: "))
        rpm = float(input("Enter RPM: "))

    except ValueError:
        print("Invalid input. Enter numeric values.")
        return

    values = [
        temperature,
        pressure,
        vibration,
        current,
        rpm,
    ]

    if not np.isfinite(values).all():
        print("Sensor values must be finite numbers.")
        return

    new_data = pd.DataFrame(
        [values],
        columns=FEATURE_NAMES,
    )

    prediction = model.predict(new_data)[0]

    print("\nPredicted machine condition:", prediction)

    print("\nClass probabilities:")

    probabilities = model.predict_proba(new_data)[0]

    for label, probability in zip(
        model.classes_,
        probabilities,
    ):
        print(f"{label}: {probability * 100:.2f}%")


# ==========================================================
# STEP 9: MAIN PROGRAM
# ==========================================================

def main():

    X, y = load_data()

    model, X_test, y_test = train_model(X, y)

    evaluate_model(model, X_test, y_test)

    display_rules(model)

    save_tree(model)

    save_model(model)

    predict_machine(model)


if __name__ == "__main__":
    main()
