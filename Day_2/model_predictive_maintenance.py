
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text, plot_tree
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

MODEL_FILE = BASE_DIR / "random_forest_model.pkl"
TREE_IMAGE = BASE_DIR / "random_forest_tree.png"
CM_IMAGE = BASE_DIR / "random_forest_confusion_matrix.png"
RULES_FILE = BASE_DIR / "random_forest_tree_rules.txt"
IMPORTANCE_IMAGE = BASE_DIR / "feature_importance.png"

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
            f"Dataset not found: {DATA_FILE}\n"
            "Place day2_sample.csv inside the data folder."
        )

    data = pd.read_csv(
        DATA_FILE,
        encoding="utf-8-sig",
    )

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

    # Convert sensor readings into numeric values.
    X = data[FEATURE_NAMES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Clean machine-condition labels.
    y = (
        data[TARGET_COLUMN]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # Remove rows with invalid sensor values or missing labels.
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

    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print("Total rows:", len(data))
    print("Valid rows:", len(X))
    print("Invalid rows:", int((~valid).sum()))

    print("\nMachine condition distribution:")
    print(y.value_counts().to_string())

    if X.empty:
        raise ValueError(
            "No valid rows found. Check the CSV sensor values."
        )

    if y.nunique() < 2:
        raise ValueError(
            "At least two different machine conditions "
            "are required for classification."
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            "Each class needs at least two samples for "
            "a stratified train/test split."
        )

    return X, y


# ==========================================================
# STEP 3: TRAIN RANDOM FOREST MODEL
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

    print("\n" + "=" * 60)
    print("RANDOM FOREST TRAINING")
    print("=" * 60)

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Number of trees:", model.n_estimators)

    print(
        f"Training accuracy: "
        f"{model.score(X_train, y_train) * 100:.2f}%"
    )

    print(
        f"Testing accuracy: "
        f"{model.score(X_test, y_test) * 100:.2f}%"
    )

    return model, X_train, X_test, y_train, y_test


# ==========================================================
# STEP 4: EVALUATE RANDOM FOREST MODEL
# ==========================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)
    labels = model.classes_

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 60)
    print("RANDOM FOREST MODEL EVALUATION")
    print("=" * 60)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=labels,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    print("\nConfusion Matrix:")

    print(
        pd.DataFrame(
            cm,
            index=labels,
            columns=labels,
        )
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )

    display.plot(
        values_format="d",
        xticks_rotation=45,
    )

    plt.title("Random Forest - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        CM_IMAGE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    print("\nConfusion matrix saved to:", CM_IMAGE)


# ==========================================================
# STEP 5: FEATURE IMPORTANCE
# ==========================================================

def show_feature_importance(model):

    importance = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False,
    )

    print("\n" + "=" * 60)
    print("SENSOR FEATURE IMPORTANCE")
    print("=" * 60)

    print(importance.to_string(index=False))

    plt.figure(figsize=(9, 5))

    plt.barh(
        importance["Feature"],
        importance["Importance"],
    )

    plt.xlabel("Importance")
    plt.ylabel("Sensor Feature")
    plt.title("Random Forest - Feature Importance")

    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(
        IMPORTANCE_IMAGE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    print("Feature importance saved to:", IMPORTANCE_IMAGE)


# ==========================================================
# STEP 6: EXTRACT RULES FROM ONE RANDOM FOREST TREE
# ==========================================================

def save_example_tree(model):

    # Random Forest contains multiple decision trees.
    # Display one tree to make the model easier to understand.
    one_tree = model.estimators_[0]

    rules = export_text(
        one_tree,
        feature_names=FEATURE_NAMES,
        decimals=3,
    )

    print("\n" + "=" * 60)
    print("RULES FROM ONE RANDOM FOREST TREE")
    print("=" * 60)

    print(rules)

    RULES_FILE.write_text(
        rules,
        encoding="utf-8",
    )

    print("\nTree rules saved to:", RULES_FILE)

    plt.figure(figsize=(24, 12))

    plot_tree(
        one_tree,
        feature_names=FEATURE_NAMES,
        class_names=[
            str(class_name)
            for class_name in model.classes_
        ],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8,
    )

    plt.title(
        "One Random Forest Tree - First Three Levels"
    )

    plt.tight_layout()

    plt.savefig(
        TREE_IMAGE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    print("Tree visualization saved to:", TREE_IMAGE)


# ==========================================================
# STEP 7: SAVE TRAINED RANDOM FOREST MODEL
# ==========================================================

def save_model(model):

    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "class_names": list(model.classes_),
    }

    joblib.dump(
        artifact,
        MODEL_FILE,
    )

    print("\nModel saved to:", MODEL_FILE)


# ==========================================================
# STEP 8: PREDICT A NEW MACHINE CONDITION
# ==========================================================

def predict_machine(model):

    print("\n" + "=" * 60)
    print("PREDICT NEW MACHINE CONDITION")
    print("=" * 60)

    try:
        temperature = float(input("Enter temperature: "))
        pressure = float(input("Enter pressure: "))
        vibration = float(input("Enter vibration: "))
        current = float(input("Enter current: "))
        rpm = float(input("Enter RPM: "))

    except ValueError:
        print("Invalid input. Enter numeric sensor readings.")
        return

    values = [
        temperature,
        pressure,
        vibration,
        current,
        rpm,
    ]

    if not np.isfinite(values).all():
        print("All sensor values must be finite numbers.")
        return

    new_data = pd.DataFrame(
        [values],
        columns=FEATURE_NAMES,
    )

    prediction = model.predict(new_data)[0]
    probabilities = model.predict_proba(new_data)[0]

    print("\nPredicted machine condition:", prediction)

    print("\nPredicted class probabilities:")

    for label, probability in zip(
        model.classes_,
        probabilities,
    ):
        print(
            f"{label}: {probability * 100:.2f}%"
        )


# ==========================================================
# STEP 9: MAIN PROGRAM
# ==========================================================

def main():

    # Load and clean the data.
    X, y = load_data()

    # Train Random Forest.
    model, X_train, X_test, y_train, y_test = train_model(
        X,
        y,
    )

    # Evaluate model performance.
    evaluate_model(
        model,
        X_test,
        y_test,
    )

    # Identify important sensor features.
    show_feature_importance(model)

    # Visualize one tree from the Random Forest.
    save_example_tree(model)

    # Save the trained model.
    save_model(model)

    # Predict a new machine condition.
    predict_machine(model)


if __name__ == "__main__":
    main()
