import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import joblib

DATA_PATH = os.path.join("data", "synthetic", "sensors.csv")
ART_DIR = "artifacts"
FIG_DIR = os.path.join("reports", "figures")

os.makedirs(FIG_DIR, exist_ok=True)

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

    model_path = os.path.join(ART_DIR, "rf.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Random Forest model not found. Run train.py first.")

    model = joblib.load(model_path)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.decision_function(X)

    precision, recall, _ = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores)

    plt.figure()
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve (AP = {ap:.3f})")
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, "pr_curve.png")
    plt.savefig(out_path)
    print(f"Saved Precision–Recall curve to {out_path}")
