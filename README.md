# Predictive Maintenance Model – Equipment Failure Forecasting

## Overview
This project develops a predictive maintenance model capable of identifying early indicators of equipment failure using sensor data. The goal is to reduce unplanned downtime by forecasting failures ahead of time, enabling proactive maintenance scheduling.

The pipeline includes synthetic data generation, feature engineering, supervised learning models, and evaluation using industry-standard metrics such as precision-recall curves.

---

## Objectives
- Simulate realistic multivariate sensor data
- Engineer predictive rolling-window features
- Train baseline models (Logistic Regression, Random Forest)
- Evaluate failure-prediction performance
- Provide an extensible framework for real industrial datasets

---

## Project Structure
```
predictive-maintenance-model/
│
├── data/
│   └── synthetic/              # Generated sensor datasets
│
├── src/
│   ├── data/
│   │   └── generate.py         # Synthetic sensor generator
│   ├── models/
│   │   ├── train.py            # Baseline ML training
│   │   └── evaluate.py         # Metrics & PR curve
│
├── artifacts/                  # Saved trained models
└── reports/
    └── figures/                # Precision-Recall and diagnostic plots
```

---

## Methods

### **1. Data Simulation**
Sensor streams include:
- Vibration patterns  
- Thermal drift  
- Cyclical signals  
- Random noise  
- Unit-specific variability  

A failure-hazard function produces binary failure labels and a “failure-within-horizon” target for predictive modeling.

### **2. Feature Engineering**
Rolling-window features (per unit):
- Mean  
- Standard deviation  
- Trend estimators  

These features mimic real-world SCADA/IIoT preprocessing pipelines.

### **3. Models**
Two baselines are included:
- **Logistic Regression** (calibrated linear baseline)  
- **Random Forest Classifier** (non-linear, stronger baseline)

Both support:
- Time-based train/test splitting  
- Probability scoring  
- Precision-recall evaluation

---

## How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Generate Synthetic Data
```bash
python src/data/generate.py
```

### Train Models
```bash
python src/models/train.py
```

### Evaluate & Produce PR Curve
```bash
python src/models/evaluate.py
```

---

## Example Outputs
- Precision-Recall curve  
- Average precision score  
- Feature importance visualization (from Random Forest)  

*Plots will be saved to:*  
```
reports/figures/
```

---

## Key Results
- Baseline models establish a starting point for failure prediction  
- Random Forest typically outperforms Logistic Regression on synthetic drift data  
- Precision-recall analysis provides insight into early-failure detection quality  

---

## Future Improvements
- Add gradient-boosting models (XGBoost/LightGBM)
- Use survival analysis for time-to-failure predictions
- Build a simple dashboard for viewing failure probabilities
- Integrate real SCADA or WIMS datasets (same schema supported)

---

## License
MIT License
