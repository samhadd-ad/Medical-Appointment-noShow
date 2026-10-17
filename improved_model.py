"""
Improved model (post-course update by Sam Haddad)

What changed compared with analysis.py (the original course version):
1. Bug fix: WaitingDays is now computed on calendar dates.
   ScheduledDay includes a time of day while AppointmentDay is at midnight, so
   same-day appointments came out as -1 day and were dropped as "invalid".
   That removed ~38,500 appointments (about 35% of the data), and same-day
   patients are the ones who almost always show up.
2. Added a Gradient Boosting model (HistGradientBoostingClassifier).
3. Models are compared by AUC, since the classes are imbalanced (~20% no-shows).

Run:  python improved_model.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score

RANDOM_STATE = 42
DATA_FILE = "KaggleV2-May-2016.csv"


def load_and_clean(path: str = DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="AppointmentID")
    df = df[df["Age"] >= 0]

    scheduled = pd.to_datetime(df["ScheduledDay"])
    appointment = pd.to_datetime(df["AppointmentDay"])

    # FIX: compare calendar dates, not timestamps
    df["WaitingDays"] = (appointment.dt.normalize() - scheduled.dt.normalize()).dt.days
    df = df[df["WaitingDays"] >= 0]  # now only removes truly invalid rows

    df["NoShow"] = (df["No-show"] == "Yes").astype(int)
    print(f"Rows after cleaning: {len(df):,}  |  no-show rate: {df['NoShow'].mean():.1%}")
    same_day = df["WaitingDays"] == 0
    print(f"No-show rate, same-day: {df.loc[same_day, 'NoShow'].mean():.1%}  |  "
          f"booked ahead: {df.loc[~same_day, 'NoShow'].mean():.1%}")
    return df


def build_features(df: pd.DataFrame):
    X = df.drop(columns=["PatientId", "AppointmentID", "ScheduledDay",
                         "AppointmentDay", "No-show", "NoShow"])
    X = pd.get_dummies(X, columns=["Gender", "Neighbourhood"], drop_first=True)
    return X, df["NoShow"]


def main():
    df = load_and_clean()
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    scaler = StandardScaler().fit(X_train)
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight="balanced",
                                                   random_state=RANDOM_STATE), True),
        "Random Forest": (RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                 random_state=RANDOM_STATE, n_jobs=-1), False),
        "Gradient Boosting": (HistGradientBoostingClassifier(class_weight="balanced",
                                                             random_state=RANDOM_STATE), False),
    }

    plt.figure(figsize=(6, 5))
    rows = []
    for name, (model, scale) in models.items():
        Xtr = scaler.transform(X_train) if scale else X_train
        Xte = scaler.transform(X_test) if scale else X_test
        model.fit(Xtr, y_train)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, proba)
        rows.append({"Model": name, "AUC": round(auc, 3),
                     "Precision": round(precision_score(y_test, pred), 3),
                     "Recall": round(recall_score(y_test, pred), 3)})
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC {auc:.2f})")

    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random guess")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves: improved pipeline")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("plot_improved_roc.png", dpi=150)

    print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
