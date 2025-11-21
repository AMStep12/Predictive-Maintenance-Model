import os, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

DATA = "data/synthetic/sensors.csv"
ART = "artifacts"
os.makedirs(ART, exist_ok=True)

def make_features(df):
    df = df.sort_values(["unit","time"])
    for c in ["s1","s2","s3"]:
        df[f"{c}_roll_mean"] = df.groupby("unit")[c].transform(lambda s: s.rolling(25, min_periods=5).mean())
        df[f"{c}_roll_std"] = df.groupby("unit")[c].transform(lambda s: s.rolling(25, min_periods=5).std())
    df = df.dropna()
    X = df[[c for c in df.columns if c.startswith(("s","s1_","s2_","s3_")) and c not in ["sensors","s4"]]]
    y = df["y"]
    return X, y

if __name__ == "__main__":
    df = pd.read_csv(DATA)
    X, y = make_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    models = {
        "logreg": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"=== {name} ===")
        print(classification_report(y_test, y_pred, digits=3))
        joblib.dump(model, os.path.join(ART, f"{name}.joblib"))
    print(f"Saved models to {ART}/")
