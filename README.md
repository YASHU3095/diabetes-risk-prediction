# Diabetes Risk Prediction: Rural Clinic Screening & 30% Capacity Triage System

Major Machine Learning Project based on the Pima Indians Diabetes Database.

## Business Problem
A rural health clinic needs a quick screening tool to flag high-risk diabetes patients so doctors can prioritize them for full diagnostic blood testing. With limited daily laboratory testing capacity, the clinic can afford to test only **30% of daily walk-in arrivals**. This system ranks patients by predicted probability to maximize the detection of real diabetics (High Recall) while keeping reagent kit waste low.

## Project Structure
```
major project/
├── data/
│   └── diabetes.csv                     # Raw Pima Indians Diabetes Dataset
├── outputs/
│   ├── model_comparison.csv             # Comparative metrics table
│   ├── confusion_matrices/
│   │   ├── all_models_confusion_matrices.png
│   │   ├── knn_confusion_matrix.png
│   │   ├── logistic_regression_confusion_matrix.png
│   │   └── decision_tree_confusion_matrix.png
│   ├── evaluation_report.txt            # Clinical performance summary report
│   └── top30_screened_patients.csv      # Risk-ranked patient batch screening list
├── diabetes_prediction.py               # Main pipeline execution script
├── requirements.txt                     # Dependencies
└── README.md                            # Project documentation
```

## How to Run
```bash
python diabetes_prediction.py
```

## Key Findings (Tested on Holdout Test Set)
- **Dataset Size:** 768 patient records (500 Healthy, 268 Diabetic)
- **Hidden Missing Values:** 0s detected and imputed with median for `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`. Valid 0s in `Pregnancies` preserved.
- **Triage Performance at 30% Capacity:**
  - **KNN (k=9):** Recall@30% = **57.4%** (31/54 diabetics caught, 16 test kits wasted)
  - **Decision Tree:** Recall@30% = **55.6%** (30/54 diabetics caught, 17 test kits wasted)
  - **Logistic Regression:** Recall@30% = **50.0%** (27/54 diabetics caught, 20 test kits wasted)

