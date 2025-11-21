import os, numpy as np, pandas as pd

np.random.seed(42)

def make_sensors(n_rows=20000, n_units=20):
    time = np.arange(n_rows)
    unit = np.random.randint(0, n_units, size=n_rows)
    s1 = np.sin(time/50) + np.random.normal(0, 0.3, n_rows)
    s2 = np.cos(time/70) + np.random.normal(0, 0.4, n_rows)
    s3 = 0.01*time + np.random.normal(0, 0.2, n_rows)
    # base hazard increases with drifting s3 and volatility in s1/s2
    hazard = 1/(1+np.exp(-(0.8*s3 + 0.5*np.abs(s1) + 0.4*np.abs(s2) - 2.5)))
    fail_now = (np.random.rand(n_rows) < hazard*0.015).astype(int)
    # label: failure in next 100 steps
    horizon = 100
    fail_next = pd.Series(fail_now).rolling(horizon, min_periods=1).max().shift(-1).fillna(0).astype(int).values
    df = pd.DataFrame({"time": time, "unit": unit, "s1": s1, "s2": s2, "s3": s3, "fail_now": fail_now, "y": fail_next})
    return df

if __name__ == "__main__":
    outdir = "data/synthetic"
    os.makedirs(outdir, exist_ok=True)
    df = make_sensors()
    df.to_csv(os.path.join(outdir, "sensors.csv"), index=False)
    print(f"Wrote {len(df):,} rows to {os.path.join(outdir,'sensors.csv')}")
