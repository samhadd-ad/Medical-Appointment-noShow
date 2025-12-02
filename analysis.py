"""
ITEC 3040 Final Project - Group G
Dataset: Medical Appointment No-Shows (KaggleV2-May-2016.csv)

This script:
1. Loads and cleans the dataset
2. Performs basic EDA (printed + plots)
3. Builds features (including WaitingDays and IsHolidayWeek)
4. Trains three classification models:
   - Logistic Regression
   - Decision Tree
   - Random Forest
5. Evaluates models (Accuracy, Precision, Recall, F1, AUC, Confusion Matrix)
6. Compares models and selects the best one (by F1-score, for reporting)
7. Saves for the UI (app.py):
   - scaler.pkl
   - best_model.pkl  (Logistic Regression for UI)
   - feature_columns.pkl
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import holidays

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from scipy.stats import chi2_contingency

RANDOM_STATE = 42
DATA_FILE = "KaggleV2-May-2016.csv"

# ===================== 1. Load data =====================

def load_data(path: str = DATA_FILE) -> pd.DataFrame:
    """Load dataset"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}. Place CSV in this folder.")
    df = pd.read_csv(path)
    print("Raw data shape:", df.shape)
    return df

# ===================== 2. Basic EDA =====================

def basic_eda(df: pd.DataFrame) -> None:
    """Print info + draw a few basic plots on raw data"""
    print("\n=== BASIC INFO ===")
    print(df.info())

    print("\n=== FIRST 5 ROWS ===")
    print(df.head())

    print("\n=== NO-SHOW VALUE COUNTS ===")
    print(df["No-show"].value_counts())

    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(df.describe())

    # ---- Plot 1: Age distribution (raw) ----
    plt.figure(figsize=(6, 4))
    df["Age"].hist(bins=40)
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.title("Age Distribution (Raw)")
    plt.tight_layout()
    plt.savefig("plot_raw_age_distribution.png", dpi=150)
    plt.show()

    # ---- Plot 2: No-show bar chart (raw) ----
    plt.figure(figsize=(5, 4))
    df["No-show"].value_counts().plot(kind="bar")
    plt.title("No-show Distribution (Raw)")
    plt.xlabel("No-show")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("plot_raw_noshow_bar.png", dpi=150)
    plt.show()

# ===================== 3. Holiday Feature =====================

def add_holiday_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary column 'IsHolidayWeek' if AppointmentDay is within ±7 days of Brazilian holiday"""
    br_holidays = holidays.Brazil(years=[2016])
    holiday_dates = [hd for hd in br_holidays.keys()]

    def is_near_holiday(appointment_day):
        if isinstance(appointment_day, pd.Timestamp) and appointment_day.tzinfo is not None:
            appointment_day = appointment_day.tz_localize(None)
        return int(any(abs((appointment_day.date() - hd).days) <= 7 for hd in holiday_dates))

    df["IsHolidayWeek"] = df["AppointmentDay"].apply(is_near_holiday)
    return df

# ===================== 4. Preprocess =====================

def preprocess_data(df: pd.DataFrame):
    """Clean data & build model features"""
    data = df.copy()

    # Drop duplicate appointments
    data = data.drop_duplicates(subset="AppointmentID")

    # Remove invalid ages
    data = data[data["Age"] >= 0]

    # Parse datetime columns
    data["ScheduledDay"] = pd.to_datetime(data["ScheduledDay"])
    data["AppointmentDay"] = pd.to_datetime(data["AppointmentDay"])

    # Compute waiting days
    data["WaitingDays"] = (data["AppointmentDay"] - data["ScheduledDay"]).dt.days

    # Remove negative waiting days
    data = data[data["WaitingDays"] >= 0]

    # Encode target: Yes -> 1 (missed), No -> 0 (attended)
    data["NoShow"] = data["No-show"].map({"Yes": 1, "No": 0})

    # ---- Add holiday feature ----
    data = add_holiday_feature(data)

    print("\nCleaned data shape:", data.shape)
    print("WaitingDays min/max:", data["WaitingDays"].min(), data["WaitingDays"].max())
    print("NoShow value counts:\n", data["NoShow"].value_counts())
    print("IsHolidayWeek value counts:\n", data["IsHolidayWeek"].value_counts())

    # ---- Plot 3: WaitingDays distribution (<= 60 days) ----
    plt.figure(figsize=(6, 4))
    data[data["WaitingDays"] <= 60]["WaitingDays"].hist(bins=30)
    plt.xlabel("Waiting Days (<= 60)")
    plt.ylabel("Count")
    plt.title("Waiting Days Distribution (Cleaned)")
    plt.tight_layout()
    plt.savefig("plot_clean_waitingdays_leq60.png", dpi=150)
    plt.show()

    # ---- Plot 4: No-show vs SMS_received (countplot) ----
    plt.figure(figsize=(6, 4))
    sns.countplot(data=data, x="SMS_received", hue="No-show")
    plt.title("No-show vs SMS_received")
    plt.xlabel("SMS_received")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("plot_clean_sms_vs_noshow.png", dpi=150)
    plt.show()

    # ---- Plot 5: No-show vs IsHolidayWeek with percentages + chi-square test ----
    plt.figure(figsize=(6, 4))
    counts = data.groupby(["IsHolidayWeek", "NoShow"]).size().unstack(fill_value=0)
    ax = plt.gca()
    colors = ["green", "red"]
    labels = ["Attended (No)", "Missed (Yes)"]
    counts.plot(kind="bar", stacked=True, color=colors, ax=ax)

    for i, x in enumerate(counts.index):
        total = counts.loc[x].sum()
        bottom = 0
        for j, col in enumerate(counts.columns):
            count = counts.loc[x, col]
            pct = 100 * count / total
            ax.text(
                i,
                bottom + count / 2,
                f"{pct:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )
            bottom += count

    # Chi-square test
    contingency = pd.crosstab(data["IsHolidayWeek"], data["NoShow"])
    chi2, p, dof, expected = chi2_contingency(contingency)
    print("\nChi-square test for IsHolidayWeek vs NoShow:")
    print("Chi2 =", chi2, "p-value =", p)

    plt.title(f"No-show vs Appointment Within 7 Days of Holiday\n(p-value={p:.3f})")
    plt.xlabel("Appointment Within 7 Days of Holiday (0=No, 1=Yes)")
    plt.ylabel("Number of Appointments")
    plt.xticks(rotation=0)
    plt.legend(labels, loc="upper right")
    plt.tight_layout()
    plt.savefig("plot_noshow_vs_near_holiday.png", dpi=150)
    plt.show()

    # ---- Remaining preprocessing for model ----
    drop_cols = [
        "PatientId",
        "AppointmentID",
        "ScheduledDay",
        "AppointmentDay",
        "No-show",
    ]
    data_model = data.drop(columns=drop_cols)

    # One-hot encode categorical features
    data_encoded = pd.get_dummies(
        data_model,
        columns=["Gender", "Neighbourhood"],
        drop_first=True,
    )

    y = data_encoded["NoShow"]
    X = data_encoded.drop(columns=["NoShow"])
    feature_columns = X.columns.tolist()
    print("\nNumber of features:", len(feature_columns))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_columns

# ===================== 5. Model evaluation =====================

def evaluate_model(name, model, Xtr, Xte, ytr, yte):
    """Train model & print metrics"""
    model.fit(Xtr, ytr)
    y_pred = model.predict(Xte)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, y_proba)
    else:
        auc = np.nan

    print(f"\n==== {name} ====")
    print("Accuracy :", round(accuracy_score(yte, y_pred), 4))
    print("Precision:", round(precision_score(yte, y_pred), 4))
    print("Recall   :", round(recall_score(yte, y_pred), 4))
    print("F1-score :", round(f1_score(yte, y_pred), 4))
    print("AUC      :", round(auc, 4) if not np.isnan(auc) else "N/A")

    print("\nConfusion Matrix:")
    print(confusion_matrix(yte, y_pred))
    print("\nClassification Report:")
    print(classification_report(yte, y_pred))

    return {"name": name, "model": model, "f1": f1_score(yte, y_pred)}

# ===================== 6. Main =====================

def main():
    df = load_data()
    basic_eda(df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler,
        feature_columns,
    ) = preprocess_data(df)

    # Define models
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    dt = DecisionTreeClassifier(max_depth=8, min_samples_split=50, class_weight="balanced", random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)

    # Evaluate models
    results = []
    results.append(evaluate_model("Logistic Regression", log_reg, X_train_scaled, X_test_scaled, y_train, y_test))
    results.append(evaluate_model("Decision Tree", dt, X_train, X_test, y_train, y_test))
    results.append(evaluate_model("Random Forest", rf, X_train, X_test, y_train, y_test))

    best_result = max(results, key=lambda r: r["f1"])
    print("\nBest model by F1:", best_result["name"])

    best_model = best_result["model"]

    # Feature importance for tree-based model
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feature_importance = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
        print("\nTop 20 feature importances:")
        print(feature_importance.head(20))
        plt.figure(figsize=(8, 10))
        feature_importance.head(20).sort_values().plot(kind="barh")
        plt.xlabel("Feature Importance")
        plt.title(f"Top 20 Feature Importances ({best_result['name']})")
        plt.tight_layout()
        plt.savefig("plot_feature_importance_top20.png", dpi=150)
        plt.show()

    # Save for UI
    best_model_for_ui = log_reg
    os.makedirs("models", exist_ok=True)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model_for_ui, f)
    with open("models/model_dt.pkl", "wb") as f:
        pickle.dump(dt, f)
    with open("models/model_rf.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(feature_columns, f)

    print("\n✅ Saved scaler.pkl, best_model.pkl (Logistic Regression), feature_columns.pkl")

if __name__ == "__main__":
    main()
