"""
WADI Native Training - Train Models Directly on WADI Attacks
=============================================================
Instead of cross-dataset transfer from HAI, train models directly on
WADI's actual attack patterns for better performance.

Research finding: "State-of-the-art F1-scores on WADI range 0.60-0.75"
when trained properly on WADI data.
"""

import os
import numpy as np
import pandas as pd
import time
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("="*80)
print("WADI NATIVE TRAINING - DIRECT WADI ATTACK LEARNING")
print("="*80)

# ============================================================================
# STEP 1: LOAD PREPROCESSED WADI DATA
# ============================================================================
print("\nSTEP 1: LOAD WADI DATA")
print("-"*80)

train_path = os.path.join(SCRIPT_DIR, 'processed_data', 'wadi_train_scaled.csv')
test_path = os.path.join(SCRIPT_DIR, 'processed_data', 'wadi_test_scaled.csv')
test_labels_path = os.path.join(SCRIPT_DIR, 'processed_data', 'wadi_test_labels.csv')

print("  Loading data (large files, be patient)...")
# Load with downsampling to fit in memory
X_train_normal = pd.read_csv(train_path, skiprows=lambda x: x > 0 and x % 10 != 0).values
X_test_mixed = pd.read_csv(test_path, skiprows=lambda x: x > 0 and x % 5 != 0).values
y_test_labels = pd.read_csv(test_labels_path, skiprows=lambda x: x > 0 and x % 5 != 0)

if 'Attack' in y_test_labels.columns:
    y_test_true = y_test_labels['Attack'].values
elif 'Normal/Attack' in y_test_labels.columns:
    y_test_true = (y_test_labels['Normal/Attack'] == -1).astype(int)
else:
    y_test_true = y_test_labels.values.flatten()

print(f"✓ Training data (pure normal): {X_train_normal.shape}")
print(f"✓ Test data (mixed): {X_test_mixed.shape}")
print(f"✓ Test labels: {y_test_true.shape}")
print(f"  - Normal (0): {np.sum(y_test_true==0):,} ({100*np.mean(y_test_true==0):.2f}%)")
print(f"  - Attack (1): {np.sum(y_test_true==1):,} ({100*np.mean(y_test_true==1):.2f}%)")

# ============================================================================
# STEP 2: SPLIT WADI DATA PROPERLY
# ============================================================================
print("\nSTEP 2: CREATE TRAIN/VAL/TEST SPLIT (NO DATA LEAKAGE)")
print("-"*80)

# Separate test data into normal and attack
test_normal_mask = (y_test_true == 0)
test_attack_mask = (y_test_true == 1)

X_test_normal = X_test_mixed[test_normal_mask]
X_test_attack = X_test_mixed[test_attack_mask]
y_test_normal = np.zeros(len(X_test_normal))
y_test_attack = np.ones(len(X_test_attack))

print(f"✓ Separated test data:")
print(f"  - Test normal: {X_test_normal.shape[0]:,} samples")
print(f"  - Test attacks: {X_test_attack.shape[0]:,} samples")

# Use 50% of test normal for validation, 50% for final test
X_val_normal, X_final_test_normal, y_val_normal, y_final_test_normal = \
    train_test_split(X_test_normal, y_test_normal, test_size=0.5, random_state=42)

# Use 70% of test attacks for validation, 30% for final test
X_val_attack, X_final_test_attack, y_val_attack, y_final_test_attack = \
    train_test_split(X_test_attack, y_test_attack, test_size=0.3, random_state=42)

# Create train/val/test sets
X_train = X_train_normal
y_train = np.zeros(len(X_train_normal))

X_val = np.vstack([X_val_normal, X_val_attack])
y_val = np.hstack([y_val_normal, y_val_attack])

X_final_test = np.vstack([X_final_test_normal, X_final_test_attack])
y_final_test = np.hstack([y_final_test_normal, y_final_test_attack])

print(f"\n✓ Train set: {X_train.shape[0]:,} samples (normal only)")
print(f"✓ Validation set: {X_val.shape[0]:,} samples")
print(f"  - Normal: {np.sum(y_val==0):,}")
print(f"  - Attack: {np.sum(y_val==1):,}")
print(f"✓ Final test set: {X_final_test.shape[0]:,} samples")
print(f"  - Normal: {np.sum(y_final_test==0):,}")
print(f"  - Attack: {np.sum(y_final_test==1):,}")

# ============================================================================
# STEP 3: TRAIN MODELS FOR ANOMALY DETECTION
# ============================================================================
print("\nSTEP 3: TRAIN ANOMALY DETECTION MODELS")
print("-"*80)

results = []

# Approach 1: Isolation Forest (detect outliers from training data)
print("\n[1] ISOLATION FOREST (Outlier Detection)")
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
iso_forest.fit(X_train)

# Get anomaly scores on validation to set threshold
val_scores = -iso_forest.score_samples(X_val)  # negative because lower is more anomalous
threshold_999 = np.percentile(val_scores, 99.9)

print(f"  Threshold (99.9th percentile): {threshold_999:.4f}")

val_pred = (val_scores > threshold_999).astype(int)
val_f1 = f1_score(y_val, val_pred)
print(f"  Validation F1: {val_f1:.4f}")

# Test on final test set
test_scores = -iso_forest.score_samples(X_final_test)
test_pred = (test_scores > threshold_999).astype(int)

tn, fp, fn, tp = confusion_matrix(y_final_test, test_pred).ravel()
f1 = f1_score(y_final_test, test_pred)
precision = precision_score(y_final_test, test_pred)
recall = recall_score(y_final_test, test_pred)
accuracy = accuracy_score(y_final_test, test_pred)

print(f"  Test Results:")
print(f"    F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
print(f"    Accuracy: {accuracy:.4f} | FP Rate: {100*fp/(fp+tn):.2f}%")

results.append({
    'Model': 'IsolationForest',
    'F1_Score': f1,
    'Precision': precision,
    'Recall': recall,
    'Accuracy': accuracy,
    'FP_Rate': 100*fp/(fp+tn),
    'TP': tp,
    'FP': fp,
    'TN': tn,
    'FN': fn
})

# Approach 2: Local Outlier Factor
print("\n[2] LOCAL OUTLIER FACTOR")
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1)
lof.fit(X_train)

# Get scores on validation
val_scores_lof = -lof.score_samples(X_val)
threshold_lof = np.percentile(val_scores_lof, 99.9)

print(f"  Threshold (99.9th percentile): {threshold_lof:.4f}")

val_pred_lof = (val_scores_lof > threshold_lof).astype(int)
val_f1_lof = f1_score(y_val, val_pred_lof)
print(f"  Validation F1: {val_f1_lof:.4f}")

# Test
test_scores_lof = -lof.score_samples(X_final_test)
test_pred_lof = (test_scores_lof > threshold_lof).astype(int)

tn, fp, fn, tp = confusion_matrix(y_final_test, test_pred_lof).ravel()
f1 = f1_score(y_final_test, test_pred_lof)
precision = precision_score(y_final_test, test_pred_lof)
recall = recall_score(y_final_test, test_pred_lof)
accuracy = accuracy_score(y_final_test, test_pred_lof)

print(f"  Test Results:")
print(f"    F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
print(f"    Accuracy: {accuracy:.4f} | FP Rate: {100*fp/(fp+tn):.2f}%")

results.append({
    'Model': 'LocalOutlierFactor',
    'F1_Score': f1,
    'Precision': precision,
    'Recall': recall,
    'Accuracy': accuracy,
    'FP_Rate': 100*fp/(fp+tn),
    'TP': tp,
    'FP': fp,
    'TN': tn,
    'FN': fn
})

# Approach 3: One-Class SVM
print("\n[3] ONE-CLASS SVM")
from sklearn.svm import OneClassSVM

oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
oc_svm.fit(X_train)

val_scores_svm = -oc_svm.decision_function(X_val)
threshold_svm = np.percentile(val_scores_svm, 99.9)

print(f"  Threshold (99.9th percentile): {threshold_svm:.4f}")

val_pred_svm = (val_scores_svm > threshold_svm).astype(int)
val_f1_svm = f1_score(y_val, val_pred_svm)
print(f"  Validation F1: {val_f1_svm:.4f}")

test_scores_svm = -oc_svm.decision_function(X_final_test)
test_pred_svm = (test_scores_svm > threshold_svm).astype(int)

tn, fp, fn, tp = confusion_matrix(y_final_test, test_pred_svm).ravel()
f1 = f1_score(y_final_test, test_pred_svm)
precision = precision_score(y_final_test, test_pred_svm)
recall = recall_score(y_final_test, test_pred_svm)
accuracy = accuracy_score(y_final_test, test_pred_svm)

print(f"  Test Results:")
print(f"    F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
print(f"    Accuracy: {accuracy:.4f} | FP Rate: {100*fp/(fp+tn):.2f}%")

results.append({
    'Model': 'OneClassSVM',
    'F1_Score': f1,
    'Precision': precision,
    'Recall': recall,
    'Accuracy': accuracy,
    'FP_Rate': 100*fp/(fp+tn),
    'TP': tp,
    'FP': fp,
    'TN': tn,
    'FN': fn
})

# Approach 4: Autoencoder-based Detection (reconstruction error)
print("\n[4] RANDOM FOREST ON EXTRACTED FEATURES")
print("  (Using PCA for dimensionality reduction)")
from sklearn.decomposition import PCA

# Use PCA to extract features
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_final_test_pca = pca.transform(X_final_test)

print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# Train RF on reconstruction error
X_train_reconstructed = pca.inverse_transform(X_train_pca)
train_recon_error = np.mean((X_train - X_train_reconstructed) ** 2, axis=1)

# Set threshold on 95th percentile of training errors
threshold_recon = np.percentile(train_recon_error, 95)
print(f"  Reconstruction error threshold: {threshold_recon:.4f}")

# Validate
X_val_reconstructed = pca.inverse_transform(X_val_pca)
val_recon_error = np.mean((X_val - X_val_reconstructed) ** 2, axis=1)
val_pred_recon = (val_recon_error > threshold_recon).astype(int)
val_f1_recon = f1_score(y_val, val_pred_recon)
print(f"  Validation F1: {val_f1_recon:.4f}")

# Test
X_final_test_reconstructed = pca.inverse_transform(X_final_test_pca)
test_recon_error = np.mean((X_final_test - X_final_test_reconstructed) ** 2, axis=1)
test_pred_recon = (test_recon_error > threshold_recon).astype(int)

tn, fp, fn, tp = confusion_matrix(y_final_test, test_pred_recon).ravel()
f1 = f1_score(y_final_test, test_pred_recon)
precision = precision_score(y_final_test, test_pred_recon)
recall = recall_score(y_final_test, test_pred_recon)
accuracy = accuracy_score(y_final_test, test_pred_recon)

print(f"  Test Results:")
print(f"    F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
print(f"    Accuracy: {accuracy:.4f} | FP Rate: {100*fp/(fp+tn):.2f}%")

results.append({
    'Model': 'PCA_ReconstructionError',
    'F1_Score': f1,
    'Precision': precision,
    'Recall': recall,
    'Accuracy': accuracy,
    'FP_Rate': 100*fp/(fp+tn),
    'TP': tp,
    'FP': fp,
    'TN': tn,
    'FN': fn
})

# ============================================================================
# STEP 4: SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("STEP 4: RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)
results_path = 'Results/WADI/WADI_NATIVE_TRAINING_RESULTS.csv'
results_df.to_csv(results_path, index=False)
print(f"\n✓ Results saved: {results_path}")

# Print summary
print("\nPerformance Comparison:")
print("-"*80)
for _, row in results_df.iterrows():
    print(f"\n{row['Model']}:")
    print(f"  F1: {row['F1_Score']:.4f} | Precision: {row['Precision']:.4f} | Recall: {row['Recall']:.4f}")
    print(f"  Accuracy: {row['Accuracy']:.4f} | FP Rate: {row['FP_Rate']:.2f}%")
    print(f"  TP: {row['TP']:,} | FP: {row['FP']:,} | TN: {row['TN']:,} | FN: {row['FN']:,}")

best_idx = results_df['F1_Score'].idxmax()
best_model = results_df.loc[best_idx]
print(f"\n{'='*80}")
print(f"🏆 BEST MODEL: {best_model['Model']}")
print(f"   F1-Score: {best_model['F1_Score']:.4f}")
print(f"{'='*80}")

print("\n✓ Analysis complete!")
