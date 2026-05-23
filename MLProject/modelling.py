import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path",        type=str, default="MLProject/social_media_mental_health_preprocessing/train.csv")
    parser.add_argument("--test_path",         type=str, default="MLProject/social_media_mental_health_preprocessing/test.csv")
    parser.add_argument("--n_estimators",      type=int, default=100)
    parser.add_argument("--max_depth",         type=str, default="None")
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf",  type=int, default=1)
    return parser.parse_args()

TARGET_COL = "PHQ_9_Severity"

def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    X_train = df_train.drop(columns=[TARGET_COL])
    y_train = df_train[TARGET_COL]
    X_test  = df_test.drop(columns=[TARGET_COL])
    y_test  = df_test[TARGET_COL]
    print(f"[load_data] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def save_confusion_matrix(y_test, y_pred, path="confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(set(y_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()

def save_feature_importance(model, feature_names, path="feature_importance.png"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[indices], color="steelblue", edgecolor="black")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.title("Feature Importances")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()

def main():
    args = parse_args()
    max_depth = None if args.max_depth == "None" else int(args.max_depth)

    X_train, X_test, y_train, y_test = load_data(args.train_path, args.test_path)

    # Log parameter ke active run
    mlflow.log_param("n_estimators",      args.n_estimators)
    mlflow.log_param("max_depth",         max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_param("min_samples_leaf",  args.min_samples_leaf)
    mlflow.log_param("random_state",      42)

    # Train
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Prediksi
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    # Metrik
    train_acc      = accuracy_score(y_train, y_train_pred)
    test_acc       = accuracy_score(y_test,  y_test_pred)
    test_f1        = f1_score(y_test,  y_test_pred, average="weighted")
    test_precision = precision_score(y_test,  y_test_pred, average="weighted", zero_division=0)
    test_recall    = recall_score(y_test,  y_test_pred, average="weighted")

    mlflow.log_metric("training_accuracy_score", train_acc)
    mlflow.log_metric("test_accuracy_score",     test_acc)
    mlflow.log_metric("test_f1_score",           test_f1)
    mlflow.log_metric("test_precision_score",    test_precision)
    mlflow.log_metric("test_recall_score",       test_recall)

    # Artefak
    save_confusion_matrix(y_test, y_test_pred, "confusion_matrix.png")
    save_feature_importance(model, list(X_train.columns), "feature_importance.png")

    cr_text = classification_report(y_test, y_test_pred, zero_division=0)
    with open("classification_report.txt", "w") as f:
        f.write(cr_text)

    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("feature_importance.png")
    mlflow.log_artifact("classification_report.txt")
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"\nTrain Accuracy : {train_acc:.4f}")
    print(f"Test  Accuracy : {test_acc:.4f}")
    print(f"Test  F1       : {test_f1:.4f}")
    print("\nClassification Report:\n", cr_text)
    print("\n[DONE] MLflow run selesai.")

if __name__ == "__main__":
    main()
