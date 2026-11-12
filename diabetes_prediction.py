import os
import sys
import tempfile

# Configure matplotlib temporary directory to prevent permissions warnings
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

def main():
    print("=" * 80)
    print(f"{'DIABETES RISK PREDICTION SYSTEM (RURAL CLINIC TRIAGE)':^80}")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. LOAD DATASET
    # --------------------------------------------------------------------------
    data_path = os.path.join("data", "diabetes.csv")
    if not os.path.exists(data_path):
        data_path = "diabetes.csv"
        
    if not os.path.exists(data_path):
        print(f"[ERROR] Could not find diabetes.csv at {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    total_records = len(df)
    print(f"Dataset loaded successfully from '{data_path}'.")
    print(f"Total records: {total_records}")

    n_diabetic = int((df['Outcome'] == 1).sum())
    n_healthy = int((df['Outcome'] == 0).sum())
    print(f"Class Balance: {n_healthy} Non-diabetic ({n_healthy/total_records:.1%}), "
          f"{n_diabetic} Diabetic ({n_diabetic/total_records:.1%})")

    # --------------------------------------------------------------------------
    # 2. AUDIT HIDDEN MISSING VALUES (ZEROS)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[PHASE 1] Physiological Hidden Zero Audit")
    print("-" * 80)
    
    hidden_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    zero_audit_rows = []
    
    for col in hidden_zero_cols:
        count = int((df[col] == 0).sum())
        pct = (count / total_records) * 100
        zero_audit_rows.append({'Feature': col, 'Hidden_Zeros': count, 'Percentage': pct})
        print(f"  {col:<22}: {count:>3} zeros ({pct:>5.1f}%) -> Physiologically invalid (Hidden NaN)")
    
    preg_zeros = int((df['Pregnancies'] == 0).sum())
    print(f"  {'Pregnancies':<22}: {preg_zeros:>3} zeros ({(preg_zeros/total_records)*100:>5.1f}%) -> Valid zero (Nulliparous women)")

    # Replace invalid zeros with NaN
    df_clean = df.copy()
    df_clean[hidden_zero_cols] = df_clean[hidden_zero_cols].replace(0, np.nan)

    # --------------------------------------------------------------------------
    # 3. STRATIFIED TRAIN-TEST SPLIT & PREPROCESSING PIPELINE
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[PHASE 2] Stratified Partitioning & Leakage-Free Pipeline")
    print("-" * 80)

    X = df_clean.drop(columns=['Outcome'])
    y = df_clean['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    print(f"Train split: {len(X_train)} samples (Diabetic: {int(y_train.sum())}, Healthy: {int((y_train == 0).sum())})")
    print(f"Test split:  {len(X_test)} samples (Diabetic: {int(y_test.sum())}, Healthy: {int((y_test == 0).sum())})")
    print("Data Leakage Shield: SimpleImputer(median) & StandardScaler fitted EXCLUSIVELY on Train split.")

    # --------------------------------------------------------------------------
    # 4. MODEL DEFINITIONS
    # --------------------------------------------------------------------------
    models = {
        'KNN': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(n_neighbors=9, weights='distance', metric='euclidean'))
        ]),
        'Logistic Regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='lbfgs'))
        ]),
        'Decision Tree': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42))
        ])
    }

    # --------------------------------------------------------------------------
    # 5. TRAINING & DEFAULT EVALUATION (T = 0.50)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Training models...")
    print("  1. KNN (k=9, distance-weighted)")
    print("  2. Logistic Regression (L2 regularized)")
    print("  3. Decision Tree (max_depth=4, min_samples_leaf=10)")
    print("-" * 80)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    comparison_records = []
    trained_pipelines = {}
    test_probs = {}
    conf_matrices = {}

    for name, pipeline in models.items():
        # Cross validation
        cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc').mean()
        
        # Fit on train
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        
        # Predict on holdout test
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        test_probs[name] = y_prob
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        conf_matrices[name] = cm
        
        comparison_records.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1_Score': round(f1, 4),
            'ROC_AUC': round(auc, 4),
            'CV_AUC_5Fold': round(cv_auc, 4)
        })

    df_comparison = pd.DataFrame(comparison_records)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON (Default Threshold T = 0.50)")
    print("=" * 80)
    print(f"{'Model':<22} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
    print("-" * 73)
    for _, row in df_comparison.iterrows():
        print(f"{row['Model']:<22} {row['Accuracy']:<10.2f} {row['Precision']:<11.2f} {row['Recall']:<10.2f} {row['F1_Score']:<10.2f} {row['ROC_AUC']:<10.2f}")

    # --------------------------------------------------------------------------
    # 6. BUSINESS LOGIC: 30% CLINIC CAPACITY CONSTRAINT EVALUATION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("30% CLINIC TESTING CAPACITY CONSTRAINT EVALUATION")
    print("=" * 80)
    
    n_test_patients = len(y_test)
    k_allowed_tests = int(np.ceil(0.30 * n_test_patients))
    total_diabetics_test = int(y_test.sum())

    print(f"Clinic testing capacity: 30%")
    print(f"Patients in walk-in test cohort: {n_test_patients}")
    print(f"Patients selected for full lab test: {k_allowed_tests} out of {n_test_patients}")
    print(f"Actual diabetic patients in cohort: {total_diabetics_test}")
    print("-" * 80)

    triage_records = []
    y_test_arr = y_test.values

    print(f"{'Model':<22} {'Tested':<8} {'Caught(TP)':<12} {'Missed(FN)':<12} {'Wasted(FP)':<12} {'Recall@30%':<12} {'Prec@30%':<10} {'Cutoff(T)'}")
    print("-" * 100)

    for name in models.keys():
        probs = test_probs[name]
        ranked_indices = np.argsort(probs)[::-1]
        
        y_triage = np.zeros(n_test_patients, dtype=int)
        y_triage[ranked_indices[:k_allowed_tests]] = 1
        
        tn, fp, fn, tp = confusion_matrix(y_test_arr, y_triage).ravel()
        rec_30 = tp / total_diabetics_test
        prec_30 = tp / k_allowed_tests
        threshold_30 = float(probs[ranked_indices[k_allowed_tests - 1]])
        
        triage_records.append({
            'Model': name,
            'Kits_Used': k_allowed_tests,
            'Diabetics_Caught_TP': int(tp),
            'Diabetics_Missed_FN': int(fn),
            'Kits_Wasted_FP': int(fp),
            'Healthy_Sent_Home_TN': int(tn),
            'Recall_at_30pct': round(rec_30, 4),
            'Precision_at_30pct': round(prec_30, 4),
            'Cutoff_Threshold': round(threshold_30, 4)
        })
        
        print(f"{name:<22} {k_allowed_tests:<8} {tp:<12} {fn:<12} {fp:<12} {rec_30:<12.1%} {prec_30:<10.1%} p >= {threshold_30:.3f}")

    df_triage = pd.DataFrame(triage_records)
    best_model_row = df_triage.sort_values(by=['Recall_at_30pct', 'Precision_at_30pct'], ascending=False).iloc[0]
    best_model_name = best_model_row['Model']

    print("-" * 100)
    print(f"Winning Model: {best_model_name}")
    print(f"Recall at 30% capacity: {best_model_row['Recall_at_30pct']:.1%} "
          f"({best_model_row['Diabetics_Caught_TP']}/{total_diabetics_test} actual diabetic patients detected)")
    print(f"Kits wasted on healthy patients: {best_model_row['Kits_Wasted_FP']} kits ({best_model_row['Precision_at_30pct']:.1%} testing precision)")

    # --------------------------------------------------------------------------
    # 7. EXPORT OUTPUTS
    # --------------------------------------------------------------------------
    outputs_dir = "outputs"
    cm_dir = os.path.join(outputs_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    # A. Save model comparison CSV
    df_combined = df_comparison.merge(df_triage, on='Model')
    comp_csv_path = os.path.join(outputs_dir, "model_comparison.csv")
    df_combined.to_csv(comp_csv_path, index=False)
    print(f"\n[Saved] Model comparison table -> '{comp_csv_path}'")

    # B. Save individual confusion matrix plots
    plt.style.use('default')
    for name, cm in conf_matrices.items():
        plt.figure(figsize=(5, 4), dpi=150)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Non-Diabetic (0)', 'Diabetic (1)'],
                    yticklabels=['Non-Diabetic (0)', 'Diabetic (1)'])
        plt.title(f'Confusion Matrix: {name} (T = 0.50)', fontsize=11, fontweight='bold', pad=12)
        plt.xlabel('Predicted Label', fontweight='semibold')
        plt.ylabel('Actual Label', fontweight='semibold')
        plt.tight_layout()
        
        file_slug = name.lower().replace(" ", "_")
        fig_path = os.path.join(cm_dir, f"{file_slug}_confusion_matrix.png")
        plt.savefig(fig_path)
        plt.close()

    # C. Save 3-panel comparative confusion matrix plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    colors = ['Blues', 'Greens', 'Purples']
    for idx, (name, cm) in enumerate(conf_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap=colors[idx], cbar=False, ax=axes[idx],
                    xticklabels=['Healthy (0)', 'Diabetic (1)'],
                    yticklabels=['Healthy (0)', 'Diabetic (1)'])
        axes[idx].set_title(f'{name}\nDefault T=0.50', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label')
        if idx == 0:
            axes[idx].set_ylabel('Actual Label')
        else:
            axes[idx].set_ylabel('')

    plt.suptitle('Clinical Confusion Matrix Comparison Across Models', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    multi_fig_path = os.path.join(cm_dir, "all_models_confusion_matrices.png")
    plt.savefig(multi_fig_path, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Confusion matrix visualizations -> '{cm_dir}'")

    # D. Save batch patient triage ranking CSV
    best_probs = test_probs[best_model_name]
    patient_ranking_df = X_test.copy()
    patient_ranking_df['Actual_Outcome'] = y_test.values
    patient_ranking_df['Predicted_Risk_Prob'] = np.round(best_probs, 4)
    patient_ranking_df = patient_ranking_df.sort_values(by='Predicted_Risk_Prob', ascending=False).reset_index(drop=True)
    patient_ranking_df['Selected_For_Lab_Test'] = 0
    patient_ranking_df.loc[:k_allowed_tests - 1, 'Selected_For_Lab_Test'] = 1
    patient_ranking_df['Screening_Rank'] = range(1, len(patient_ranking_df) + 1)

    triage_csv_path = os.path.join(outputs_dir, "top30_screened_patients.csv")
    patient_ranking_df.to_csv(triage_csv_path, index=False)
    print(f"[Saved] Patient batch screening rankings -> '{triage_csv_path}'")

    # E. Save comprehensive evaluation text report
    report_txt_path = os.path.join(outputs_dir, "evaluation_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RURAL HEALTH CLINIC - DIABETES RISK PREDICTION EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset Size: {total_records} records (500 Healthy, 268 Diabetic)\n")
        f.write(f"Train/Test Partition: 80% Train ({len(X_train)}), 20% Holdout Test ({len(X_test)})\n")
        f.write("Missing Value Treatment: Zeros in Glucose, BP, Skin, Insulin, BMI replaced with NaN\n")
        f.write("Imputation: Median imputation fitted strictly on Train split (Zero Leakage)\n\n")
        f.write("1. DEFAULT THRESHOLD PERFORMANCE (T = 0.50)\n")
        f.write("-" * 80 + "\n")
        f.write(df_comparison.to_string(index=False) + "\n\n")
        f.write("2. 30% CLINIC CAPACITY CONSTRAINT EVALUATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Testing Capacity: 30% of daily arrivals ({k_allowed_tests} tests permitted out of {n_test_patients})\n")
        f.write(f"Diabetics in Test Cohort: {total_diabetics_test}\n\n")
        f.write(df_triage.to_string(index=False) + "\n\n")
        f.write("3. CLINICAL RECOMMENDATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Recommended Winning Model: {best_model_name}\n")
        f.write(f"Justification: Captures {best_model_row['Recall_at_30pct']:.1%} of real diabetic patients "
                f"({best_model_row['Diabetics_Caught_TP']}/{total_diabetics_test}) while utilizing only "
                f"{k_allowed_tests} test kits.\n")
        f.write(f"Cutoff Threshold: Patients with risk score >= {best_model_row['Cutoff_Threshold']:.3f} "
                f"are immediately queued for phlebotomy.\n")
        f.write("Remaining patients receive non-invasive dietary/exercise guidance and 6-month checkup.\n")

    print(f"[Saved] Full evaluation report -> '{report_txt_path}'")
    print("\n" + "=" * 80)
    print("IMPLEMENTATION PLAN EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()

