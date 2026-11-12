"""
================================================================================
PHASE 5: TESTING & VALIDATION WITH SAMPLE DAILY WALK-IN DATA
Tests patient triage ranking on data/sample_daily_walkins.csv
================================================================================
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def run_sample_batch_test():
    print("=" * 80)
    print(f"{'PHASE 5: DAILY WALK-IN BATCH TESTING & CLINIC TRIAGE DEMO':^80}")
    print("=" * 80)

    # 1. Load training data to train winning model
    data_path = os.path.join("data", "diabetes.csv")
    if not os.path.exists(data_path):
        data_path = "diabetes.csv"

    df = pd.read_csv(data_path)
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_clean = df.copy()
    df_clean[zero_cols] = df_clean[zero_cols].replace(0, np.nan)

    X = df_clean.drop(columns=['Outcome'])
    y = df_clean['Outcome']

    # Train winning KNN and Logistic Regression models
    knn_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(n_neighbors=9, weights='uniform', metric='euclidean'))
    ])
    knn_pipeline.fit(X, y)

    logreg_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])
    logreg_pipeline.fit(X, y)

    # 2. Load daily walk-in sample cohort
    sample_path = os.path.join("data", "sample_daily_walkins.csv")
    if not os.path.exists(sample_path):
        print(f"[ERROR] Sample file not found at {sample_path}")
        return

    walkins_df = pd.read_csv(sample_path)
    total_walkins = len(walkins_df)
    print(f"Sample cohort loaded: {total_walkins} patients arrived at clinic today.")
    
    # 3. Calculate 30% testing quota
    quota = int(np.ceil(0.30 * total_walkins))
    print(f"Daily Phlebotomy Testing Quota (30%): {quota} patients permitted full blood test.")
    print("-" * 80)

    # 4. Predict risk probability for each patient
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X_walkins = walkins_df[features].copy()
    X_walkins[zero_cols] = X_walkins[zero_cols].replace(0, np.nan)

    knn_probs = knn_pipeline.predict_proba(X_walkins)[:, 1]
    lr_probs = logreg_pipeline.predict_proba(X_walkins)[:, 1]

    walkins_df['KNN_Risk_Prob'] = np.round(knn_probs, 4)
    walkins_df['LogReg_Risk_Prob'] = np.round(lr_probs, 4)
    walkins_df['Risk_Percentage'] = [f"{p:.1%}" for p in lr_probs]

    # 5. Rank descending by risk (using calibrated Logistic Regression probabilities)
    ranked_df = walkins_df.sort_values(by='LogReg_Risk_Prob', ascending=False).reset_index(drop=True)
    ranked_df['Triage_Rank'] = range(1, total_walkins + 1)
    
    # 6. Assign clinical decision
    ranked_df['Clinical_Action'] = 'Routine Lifestyle Advice (Home)'
    ranked_df.loc[:quota - 1, 'Clinical_Action'] = '🚨 PRIORITY LAB BLOOD TEST'

    print(f"{'Rank':<6} {'Patient ID':<12} {'Risk %':<10} {'Glucose':<10} {'BMI':<8} {'Age':<6} {'Action Assigned'}")
    print("-" * 80)
    for _, row in ranked_df.iterrows():
        action_color = row['Clinical_Action']
        print(f"{row['Triage_Rank']:<6} {row['Patient_ID']:<12} {row['Risk_Percentage']:<10} {row['Glucose']:<10} {row['BMI']:<8} {row['Age']:<6} {action_color}")

    # 7. Save output results
    output_path = os.path.join("outputs", "monday_morning_triage_results.csv")
    ranked_df.to_csv(output_path, index=False)
    print("-" * 80)
    print(f"[SUCCESS] Triage ranking sheet exported to '{output_path}'")
    print(f"Operating Threshold: Patients with predicted risk score >= {ranked_df.loc[quota - 1, 'Risk_Percentage']} receive tests.")
    print("=" * 80)

if __name__ == "__main__":
    run_sample_batch_test()
