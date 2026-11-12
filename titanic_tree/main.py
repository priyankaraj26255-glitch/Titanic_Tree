"""Titanic survival prediction with a Decision Tree.

Steps:
  1. Load and preprocess Titanic data
  2. Fit a DecisionTreeClassifier
  3. Visualize the tree using plot_tree
  4. Evaluate accuracy on the test set
  5. Export the trained model
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "titanic.csv"
MODEL_PATH = BASE / "models" / "titanic_tree.joblib"
PLOT_PATH = BASE / "outputs" / "decision_tree.png"

FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]


# 1. Load and preprocess -----------------------------------------------------
def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)
    df = df[FEATURES + ["survived"]].copy()

    df["age"] = df["age"].fillna(df["age"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["embarked"], prefix="emb", dtype=int)

    X = df.drop(columns="survived")
    y = df["survived"]
    return X, y


def main():
    X, y = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 2. Fit ------------------------------------------------------------------
    # max_depth limits overfitting and keeps the plot readable
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    # 3. Visualize ------------------------------------------------------------
    PLOT_PATH.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(22, 10))
    plot_tree(
        model,
        feature_names=X.columns,
        class_names=["Died", "Survived"],
        filled=True,
        rounded=True,
        fontsize=9,
    )
    plt.title("Titanic Survival - Decision Tree (max_depth=4)")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"Tree plot saved to {PLOT_PATH}")

    # 4. Evaluate -------------------------------------------------------------
    preds = model.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, preds):.4f}\n")
    print(classification_report(y_test, preds, target_names=["Died", "Survived"]))

    # 5. Export ---------------------------------------------------------------
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "columns": list(X.columns)}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
