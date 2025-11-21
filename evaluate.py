import os, pandas as pd, matplotlib.pyplot as plt, joblib
from sklearn.metrics import precision_recall_curve, average_precision_score

DATA = "data/synthetic/sensors.csv"
ART = "artifacts"
FIG = "reports/figures"
os.makedirs(FIG, exist_ok=True)

def load_features():
    df = pd.read_csv(DATA)
    df = df.sort_values(["unit","time"])
    for c in ["s1","s2","s3"]:
        df[f"{c}_roll_mean"] = df.groupby("unit")[c].transform(lambda s: s.rolling(25, min_periods=5).mean())
        df[f"{c}_roll_std"] = df.groupby("unit")[c].transform(lambda s: s.rolling(25, min_periods=5).std())
    df = df.dropna()
    X = df[[c for c in df.columns if c.endswith(("roll_mean","roll_std")) or c in ["s1","s2","s3"]]]
    y = df["y"]
    return X, y

if __name__ == "__main__":
    X, y = load_features()
    model = joblib.load(os.path.join(ART, "rf.joblib"))
    scores = model.predict_proba(X)[:,1]
    precision, recall, thr = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores)
    plt.figure()
    plt.step(recall, precision, where="post")
    plt.title(f"Precision-Recall (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "pr_curve.png"))
    print(f"Saved figure to {os.path.join(FIG,'pr_curve.png')}")
