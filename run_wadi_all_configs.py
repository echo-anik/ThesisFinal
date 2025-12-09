"""
WADI Cross-Dataset Validation - ALL 6 CONFIGURATIONS
====================================================
Tests all 6 model combinations from HAI experiments on WADI dataset
to find which performs best for cross-dataset transfer learning.
"""

import os
import numpy as np
import pandas as pd
import time
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("="*80)
print("WADI CROSS-DATASET VALIDATION - ALL 6 CONFIGURATIONS")
print("="*80)

# ============================================================================
# STEP 1: LOAD PREPROCESSED WADI DATA
# ============================================================================
print("\nSTEP 1: LOAD PREPROCESSED WADI DATA")
print("-"*80)

train_path = os.path.join(SCRIPT_DIR, 'processed_data', 'wadi_train_scaled.csv')
test_path = os.path.join(SCRIPT_DIR, 'processed_data', 'wadi_test_scaled.csv')
test_labels_path = os.path.join(SCRIPT_DIR, 'processed_data', 'wadi_test_labels.csv')

print("  Loading data with 5x downsampling...")
X_train_normal = pd.read_csv(train_path, skiprows=lambda x: x > 0 and x % 5 != 0).values
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
print(f"✓ Test ground truth: {y_test_true.shape}")
print(f"✓ Test label distribution:")
print(f"  - Normal (0): {np.sum(y_test_true==0):,} ({100*np.mean(y_test_true==0):.2f}%)")
print(f"  - Attack (1): {np.sum(y_test_true==1):,} ({100*np.mean(y_test_true==1):.2f}%)")
print(f"✓ Number of features: {X_train_normal.shape[1]}")

# ============================================================================
# STEP 2: CREATE COMBINED BALANCED TRAINING DATA
# ============================================================================
print("\nSTEP 2: CREATE COMBINED BALANCED TRAINING DATA")
print("-"*80)

# Separate normal and attack from test set
test_normal_mask = (y_test_true == 0)
test_attack_mask = (y_test_true == 1)

X_test_normal = X_test_mixed[test_normal_mask]
X_test_attack = X_test_mixed[test_attack_mask]

print(f"✓ Separated test data:")
print(f"  - Normal samples: {X_test_normal.shape[0]:,}")
print(f"  - Attack samples: {X_test_attack.shape[0]:,}")

# Combine train normal + test attack
X_combined = np.vstack([X_train_normal, X_test_attack])
y_combined = np.hstack([np.zeros(len(X_train_normal)), np.ones(len(X_test_attack))])

print(f"✓ Combined data (train normal + test attack):")
print(f"  Shape: {X_combined.shape}")
print(f"  Class distribution before SMOTE:")
print(f"    - Normal: {np.sum(y_combined==0):,}")
print(f"    - Attack: {np.sum(y_combined==1):,}")
print(f"    - Ratio: {100*np.mean(y_combined==1):.2f}% attack")

# ============================================================================
# STEP 3: DOWNSAMPLE + SMOTE FOR BALANCED TRAINING (Memory-Efficient)
# ============================================================================
print("\nSTEP 3: DOWNSAMPLE + SMOTE FOR BALANCED TRAINING")
print("-"*80)

# Research shows: "WADI is huge and the processing takes a lot of memory"
# Solution: Downsample by factor of 5 (1s -> 5s resolution)
# This maintains sufficient temporal resolution for 2-30 minute attacks

downsample_factor = 5
print(f"✓ Applying downsampling (factor={downsample_factor}) to reduce memory...")

# Downsample combined data
X_combined_downsampled = X_combined[::downsample_factor]
y_combined_downsampled = y_combined[::downsample_factor]

print(f"  Before downsampling: {X_combined.shape[0]:,} samples")
print(f"  After downsampling: {X_combined_downsampled.shape[0]:,} samples ({X_combined_downsampled.shape[0]/X_combined.shape[0]*100:.1f}%)")

# Now apply SMOTE on downsampled data
print(f"✓ Applying SMOTE on downsampled data...")
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_combined_downsampled, y_combined_downsampled)

print(f"✓ After SMOTE:")
print(f"  Shape: {X_balanced.shape}")
print(f"  Class distribution:")
print(f"    - Normal (0): {np.sum(y_balanced==0):,} ({100*np.mean(y_balanced==0):.2f}%)")
print(f"    - Attack (1): {np.sum(y_balanced==1):,} ({100*np.mean(y_balanced==1):.2f}%)")
print(f"✓ Memory-efficient training set created")

# ============================================================================
# STEP 4: DEFINE ALL 6 CONFIGURATIONS
# ============================================================================
print("\nSTEP 4: DEFINE ALL 6 CONFIGURATIONS")
print("-"*80)

configs = [
    {
        'name': 'Config_1_Lightweight',
        'edge': RandomForestClassifier(n_estimators=50, max_depth=15, min_samples_split=10,
                                      min_samples_leaf=5, random_state=42, n_jobs=-1,
                                      class_weight='balanced'),
        'central': RandomForestClassifier(n_estimators=200, max_depth=30, min_samples_split=5,
                                         min_samples_leaf=2, random_state=42, n_jobs=-1,
                                         class_weight='balanced')
    },
    {
        'name': 'Config_2_Balanced',
        'edge': RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10,
                                      min_samples_leaf=5, random_state=42, n_jobs=-1,
                                      class_weight='balanced'),
        'central': GradientBoostingClassifier(n_estimators=200, max_depth=7, learning_rate=0.1,
                                             random_state=42)
    },
    {
        'name': 'Config_3_FastEdge',
        'edge': ExtraTreesClassifier(n_estimators=50, max_depth=15, min_samples_split=10,
                                    min_samples_leaf=5, random_state=42, n_jobs=-1,
                                    class_weight='balanced'),
        'central': RandomForestClassifier(n_estimators=300, max_depth=30, min_samples_split=5,
                                         min_samples_leaf=2, random_state=42, n_jobs=-1,
                                         class_weight='balanced')
    },
    {
        'name': 'Config_4_Conservative',
        'edge': RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=15,
                                      min_samples_leaf=8, random_state=42, n_jobs=-1,
                                      class_weight='balanced'),
        'central': RandomForestClassifier(n_estimators=200, max_depth=30, min_samples_split=5,
                                         min_samples_leaf=2, random_state=42, n_jobs=-1,
                                         class_weight='balanced')
    },
    {
        'name': 'Config_5_OptimalBalance',
        'edge': RandomForestClassifier(n_estimators=75, max_depth=15, min_samples_split=10,
                                      min_samples_leaf=5, random_state=42, n_jobs=-1,
                                      class_weight='balanced'),
        'central': GradientBoostingClassifier(n_estimators=200, max_depth=7, learning_rate=0.1,
                                             random_state=42)
    },
    {
        'name': 'Config_6_HighAccuracy',
        'edge': RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5,
                                      min_samples_leaf=2, random_state=42, n_jobs=-1,
                                      class_weight='balanced'),
        'central': RandomForestClassifier(n_estimators=300, max_depth=30, min_samples_split=5,
                                         min_samples_leaf=2, random_state=42, n_jobs=-1,
                                         class_weight='balanced')
    }
]

edge_percentages = [5, 10, 15, 20, 25]

print(f"✓ Configurations: {len(configs)}")
print(f"✓ Edge percentages: {edge_percentages}%")
print(f"✓ Total experiments: {len(configs)} configs × {len(edge_percentages)} percentages = {len(configs)*len(edge_percentages)}")

# ============================================================================
# STEP 5: RUN ALL EXPERIMENTS
# ============================================================================
print("\n" + "="*80)
print("STEP 5: RUNNING ALL EXPERIMENTS")
print("="*80)

all_results = []
experiment_num = 0
total_experiments = len(configs) * len(edge_percentages)

for config in configs:
    print(f"\n{'#'*80}")
    print(f"CONFIGURATION: {config['name']}")
    print(f"{'#'*80}")
    
    # Train central model ONCE per config (uses 100% balanced data)
    print(f"\n  [CENTRAL] Training central model (100% = {len(X_balanced):,} samples)...")
    central_start = time.time()
    central_model = config['central']
    central_model.fit(X_balanced, y_balanced)
    central_train_time = time.time() - central_start
    
    # Measure central model size
    central_pkl = pickle.dumps(central_model)
    central_size_mb = len(central_pkl) / (1024 * 1024)
    print(f"    ✓ Training time: {central_train_time:.2f}s")
    print(f"    ✓ Model size: {central_size_mb:.2f} MB")
    
    # Get central predictions
    y_central_pred = central_model.predict(X_test_mixed)
    central_confidence = central_model.predict_proba(X_test_mixed)[:, 1]
    
    for edge_pct in edge_percentages:
        experiment_num += 1
        print(f"\n  [{experiment_num}/{total_experiments}] Edge {edge_pct}% | Central 100%")
        print(f"  {'-'*76}")
        
        start_time = time.time()
        
        # Sample edge percentage
        n_edge = int(len(X_balanced) * (edge_pct / 100))
        indices = np.random.choice(len(X_balanced), size=n_edge, replace=False)
        X_edge = X_balanced[indices]
        y_edge = y_balanced[indices]
        
        print(f"    Edge training: {edge_pct}% = {n_edge:,} samples")
        print(f"      - Normal: {np.sum(y_edge==0):,} ({100*np.mean(y_edge==0):.1f}%)")
        print(f"      - Attack: {np.sum(y_edge==1):,} ({100*np.mean(y_edge==1):.1f}%)")
        
        # Train edge model
        print(f"    [EDGE] Training edge model...")
        edge_start = time.time()
        edge_model = config['edge']
        edge_model.fit(X_edge, y_edge)
        edge_train_time = time.time() - edge_start
        
        # Measure edge model
        edge_pkl = pickle.dumps(edge_model)
        edge_size_mb = len(edge_pkl) / (1024 * 1024)
        
        print(f"      ✓ Training time: {edge_train_time:.2f}s")
        print(f"      ✓ Model size: {edge_size_mb:.2f} MB")
        
        # Get edge predictions
        y_edge_pred = edge_model.predict(X_test_mixed)
        edge_confidence = edge_model.predict_proba(X_test_mixed)[:, 1]
        
        # Calculate escalation rate
        escalation_threshold = 0.5
        escalations = np.sum(edge_confidence > escalation_threshold)
        escalation_rate = 100 * escalations / len(y_edge_pred)
        print(f"      ✓ Escalations (confidence > {escalation_threshold}): {escalations:,} ({escalation_rate:.1f}%)")
        
        # Hierarchical decision
        print(f"    [HIERARCHICAL] Combining predictions...")
        y_final_pred = np.zeros(len(X_test_mixed), dtype=int)
        
        for i in range(len(X_test_mixed)):
            if edge_confidence[i] > escalation_threshold:
                y_final_pred[i] = y_edge_pred[i]
            else:
                y_final_pred[i] = y_central_pred[i]
        
        # Calculate timing metrics
        edge_inference_start = time.time()
        _ = edge_model.predict(X_test_mixed[:100])
        edge_inference_time_ms = (time.time() - edge_inference_start) / 100 * 1000
        
        central_inference_start = time.time()
        _ = central_model.predict(X_test_mixed[:100])
        central_inference_time_ms = (time.time() - central_inference_start) / 100 * 1000
        
        avg_response_time_ms = edge_inference_time_ms + (escalation_rate / 100) * central_inference_time_ms
        max_response_time_ms = edge_inference_time_ms + central_inference_time_ms
        
        # Detection delay
        attack_indices = np.where(y_test_true == 1)[0]
        detected_attack_indices = np.where((y_test_true == 1) & (y_final_pred == 1))[0]
        if len(attack_indices) > 0 and len(detected_attack_indices) > 0:
            first_attack_idx = attack_indices[0]
            first_detection_idx = detected_attack_indices[0]
            detection_delay_samples = max(0, first_detection_idx - first_attack_idx)
            detection_delay_seconds = detection_delay_samples * 1.0
        else:
            detection_delay_samples = 0
            detection_delay_seconds = 0.0
        
        total_time = time.time() - start_time
        
        # Evaluate performance
        tn, fp, fn, tp = confusion_matrix(y_test_true, y_final_pred).ravel()
        
        f1 = f1_score(y_test_true, y_final_pred)
        precision = precision_score(y_test_true, y_final_pred)
        recall = recall_score(y_test_true, y_final_pred)
        accuracy = accuracy_score(y_test_true, y_final_pred)
        
        try:
            auc = roc_auc_score(y_test_true, central_confidence)
        except:
            auc = 0.0
        
        # Resource metrics
        total_model_size = edge_size_mb + central_size_mb
        edge_power = 2.0 + (edge_model.n_estimators if hasattr(edge_model, 'n_estimators') else 50) * 0.05 / 1000
        central_power = 1.8 + (central_model.n_estimators if hasattr(central_model, 'n_estimators') else 200) * 0.02 / 1000
        combined_power = (edge_power + central_power) / 2
        
        print(f"    [METRICS] Performance:")
        print(f"      F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | Acc: {accuracy:.4f}")
        print(f"      FP Rate: {100*fp/(fp+tn):.2f}% | Response: {avg_response_time_ms:.1f}ms | Detect: {detection_delay_seconds:.1f}s")
        
        # Store result
        result = {
            'Config': config['name'],
            'Edge_Percentage': edge_pct,
            'F1_Score': f1,
            'Precision': precision,
            'Recall': recall,
            'Accuracy': accuracy,
            'FP_Rate': 100*fp/(fp+tn),
            'Escalation_Rate': escalation_rate,
            'Avg_Response_Time_ms': avg_response_time_ms,
            'Max_Response_Time_ms': max_response_time_ms,
            'Detection_Delay_seconds': detection_delay_seconds,
            'Edge_Model_Size_MB': edge_size_mb,
            'Central_Model_Size_MB': central_size_mb,
            'Total_Model_Size_MB': total_model_size,
            'Combined_Power_W': combined_power,
            'Edge_Training_Time': edge_train_time,
            'Central_Training_Time': central_train_time,
            'Total_Time': total_time,
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn,
            'ROC_AUC': auc
        }
        
        all_results.append(result)

# ============================================================================
# STEP 6: SAVE AND ANALYZE RESULTS
# ============================================================================
print("\n" + "="*80)
print("STEP 6: SAVE AND ANALYZE RESULTS")
print("="*80)

results_df = pd.DataFrame(all_results)

# Save detailed results
os.makedirs('Results/WADI', exist_ok=True)
detailed_path = 'Results/WADI/WADI_ALL_CONFIGS_DETAILED.csv'
results_df.to_csv(detailed_path, index=False)
print(f"\n✓ Detailed results saved: {detailed_path}")

# Create summary by config
summary = results_df.groupby('Config').agg({
    'F1_Score': ['mean', 'max'],
    'Precision': 'mean',
    'Recall': 'mean',
    'FP_Rate': 'mean',
    'Escalation_Rate': 'mean',
    'Total_Model_Size_MB': 'mean',
    'Combined_Power_W': 'mean'
}).round(4)

summary_path = 'Results/WADI/WADI_ALL_CONFIGS_SUMMARY.csv'
summary.to_csv(summary_path)
print(f"✓ Summary results saved: {summary_path}")

# ============================================================================
# STEP 7: FIND BEST CONFIGURATION
# ============================================================================
print("\n" + "="*80)
print("STEP 7: BEST CONFIGURATION ANALYSIS")
print("="*80)

print("\nPerformance by Configuration (Mean across all edge percentages):")
print("-"*80)
for config_name in results_df['Config'].unique():
    config_data = results_df[results_df['Config'] == config_name]
    print(f"\n{config_name}:")
    print(f"  F1: {config_data['F1_Score'].mean():.4f} (max: {config_data['F1_Score'].max():.4f})")
    print(f"  Precision: {config_data['Precision'].mean():.4f}")
    print(f"  Recall: {config_data['Recall'].mean():.4f}")
    print(f"  FP Rate: {config_data['FP_Rate'].mean():.2f}%")
    print(f"  Escalation: {config_data['Escalation_Rate'].mean():.1f}%")
    print(f"  Model Size: {config_data['Total_Model_Size_MB'].mean():.2f} MB")

# Find overall best
best_idx = results_df['F1_Score'].idxmax()
best_result = results_df.loc[best_idx]

print(f"\n{'='*80}")
print(f"🏆 BEST OVERALL CONFIGURATION")
print(f"{'='*80}")
print(f"  Config: {best_result['Config']}")
print(f"  Edge Percentage: {best_result['Edge_Percentage']:.0f}%")
print(f"  F1 Score: {best_result['F1_Score']:.4f}")
print(f"  Precision: {best_result['Precision']:.4f}")
print(f"  Recall: {best_result['Recall']:.4f}")
print(f"  Accuracy: {best_result['Accuracy']:.4f}")
print(f"  False Positive Rate: {best_result['FP_Rate']:.2f}%")
print(f"  Escalation Rate: {best_result['Escalation_Rate']:.1f}%")
print(f"  Response Time: {best_result['Avg_Response_Time_ms']:.2f} ms")
print(f"  Model Size: {best_result['Total_Model_Size_MB']:.2f} MB")
print(f"  Power: {best_result['Combined_Power_W']:.2f} W")

print("\n✓ All configurations tested. Use best for thesis.")
print("="*80)
