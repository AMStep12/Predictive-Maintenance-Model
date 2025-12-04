import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

DATA_PATH = os.path.join("data", "synthetic", "sensors.csv")
ART_DIR = "artifacts"

os.makedirs(ART_DIR, exist_ok=True)

def make_features(df):
    df = df.sort_values(["unit", "time"]).copy()

    for c in ["s1", "s2", "s3"]:
        df[f"{c}_roll_mean"] = (
            df.groupby("unit")[c]
            .transform(lambda s: s.rolling(25, min_periods=5).mean())
        )
        df[f"{c}_roll_std"] = (
            df.groupby("unit")[c]
            .transform(lambda s: s.rolling(25, min_periods=5).std())
        )

    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "s1", "s2", "s3",
        "s1_roll_mean", "s1_roll_std",
        "s2_roll_mean", "s2_roll_std",
        "s3_roll_mean", "s3_roll_std",
    ]

    X = df[feature_cols]
    y = df["y"]
    return X, y

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    X, y = make_features(df)

    # Time-based split: keep temporal order
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    models = {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("=" * 40)
        print(f"Model: {name}")
        print(classification_report(y_test, y_pred, digits=3))
        joblib.dump(model, os.path.join(ART_DIR, f"{name}.joblib"))

    print(f"Saved models to {ART_DIR}/")
