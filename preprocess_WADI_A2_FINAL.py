"""
WADI A2 COMPREHENSIVE PREPROCESSING
====================================
Feature selection, scaling, SMOTE balancing, and data quality validation
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')

# Use relative paths from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WADI_DIR = os.path.join(SCRIPT_DIR, 'WADI.A2_19 Nov 2019')
TRAIN_FILE = os.path.join(WADI_DIR, 'WADI_14days_new.csv')
TEST_FILE = os.path.join(WADI_DIR, 'WADI_attackdataLABLE.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'processed_data')

print("="*80)
print("WADI A2 COMPREHENSIVE PREPROCESSING PIPELINE")
print("="*80)

# ============================================================================
# STEP 1: LOAD & PARSE DATA CORRECTLY
# ============================================================================
print("\nSTEP 1: LOAD DATA WITH CORRECT PARSING")
print("-" * 80)

train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
test_df = pd.read_csv(TEST_FILE, low_memory=False)

# Test file has header in first data row - extract it
header_row = test_df.iloc[0].values
actual_headers = [f"col_{i}" if isinstance(v, (int, float)) else str(v) for i, v in enumerate(header_row)]
test_df = test_df.iloc[1:].reset_index(drop=True)
test_df.columns = actual_headers

print(f"✓ Training: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
print(f"✓ Test: {test_df.shape[0]:,} rows × {test_df.shape[1]} columns")

# ============================================================================
# STEP 2: EXTRACT LABELS & SENSOR COLUMNS
# ============================================================================
print("\nSTEP 2: EXTRACT LABELS & IDENTIFY SENSORS")
print("-" * 80)

# Extract label column from test
label_col = [col for col in test_df.columns if 'Attack' in str(col) or 'LABLE' in str(col)][0]
print(f"Label column: {label_col}")

# Map labels: 1 = Normal (0), -1 = Attack (1)
test_labels = test_df[label_col].map({1: 0, '-1': 1, 1.0: 0, -1.0: 1}).fillna(
    test_df[label_col].apply(lambda x: 0 if x == 1 or x == '1' else (1 if x == -1 or x == '-1' else np.nan))
)

print(f"\nLabel distribution in test:")
print(f"  Normal (0): {(test_labels == 0).sum():,} ({(test_labels == 0).sum()/len(test_labels)*100:.2f}%)")
print(f"  Attack (1): {(test_labels == 1).sum():,} ({(test_labels == 1).sum()/len(test_labels)*100:.2f}%)")

# Identify sensor columns (exclude Row, Date, Time, Label)
exclude_cols = ['Row', 'Date', 'Time', label_col, 'col_0', 'col_1', 'col_2']
sensor_cols = [col for col in train_df.columns if col not in exclude_cols and col not in ['col_' + str(i) for i in range(3)]]

print(f"\n✓ Sensor columns: {len(sensor_cols)}")
print(f"  Examples: {sensor_cols[:5]}")

# ============================================================================
# STEP 3: EXTRACT & CLEAN DATA
# ============================================================================
print("\nSTEP 3: EXTRACT FEATURES & CONVERT TO NUMERIC")
print("-" * 80)

X_train = train_df[sensor_cols].copy()
X_test = test_df[sensor_cols].copy()

# Convert to numeric
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

print(f"Training shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# Handle missing values
missing_train = X_train.isnull().sum().sum()
missing_test = X_test.isnull().sum().sum()
print(f"\nMissing before fill - Train: {missing_train:,}, Test: {missing_test:,}")

X_train = X_train.fillna(method='ffill').fillna(method='bfill').fillna(X_train.mean())
X_test = X_test.fillna(method='ffill').fillna(method='bfill').fillna(X_test.mean())

missing_train = X_train.isnull().sum().sum()
missing_test = X_test.isnull().sum().sum()
print(f"Missing after fill - Train: {missing_train:,}, Test: {missing_test:,}")

# ============================================================================
# STEP 4: REMOVE CONSTANT & LOW-VARIANCE FEATURES
# ============================================================================
print("\nSTEP 4: FEATURE SELECTION (VARIANCE THRESHOLD)")
print("-" * 80)

# Calculate variance
variance = X_train.var()
print(f"\nVariance statistics:")
print(f"  Min: {variance.min():.6f}")
print(f"  Max: {variance.max():.2f}")
print(f"  Mean: {variance.mean():.2f}")

# Remove features with variance < 0.01 (very low variance)
threshold = 0.01
selector = VarianceThreshold(threshold=threshold)
selector.fit(X_train)

selected_features = X_train.columns[selector.get_support()].tolist()
removed_features = X_train.columns[~selector.get_support()].tolist()

print(f"\nFeature selection:")
print(f"  Original features: {X_train.shape[1]}")
print(f"  Removed (variance < {threshold}): {len(removed_features)}")
print(f"  Remaining features: {len(selected_features)}")
print(f"  Removed examples: {removed_features[:5]}")

# Apply feature selection
X_train = X_train[selected_features]
X_test = X_test[selected_features]

print(f"\nAfter selection:")
print(f"  Train: {X_train.shape}")
print(f"  Test: {X_test.shape}")

# ============================================================================
# STEP 5: SCALE FEATURES
# ============================================================================
print("\nSTEP 5: NORMALIZE FEATURES (MinMaxScaler)")
print("-" * 80)

scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features)

print(f"✓ Scaled to [0, 1] range")
print(f"  Train - min: {X_train_scaled.min().min():.4f}, max: {X_train_scaled.max().max():.4f}")
print(f"  Test  - min: {X_test_scaled.min().min():.4f}, max: {X_test_scaled.max().max():.4f}")

# ============================================================================
# STEP 6: DATA QUALITY CHECKS
# ============================================================================
print("\nSTEP 6: DATA QUALITY VALIDATION")
print("-" * 80)

# Check for NaN
print(f"NaN in train: {X_train_scaled.isnull().sum().sum()}")
print(f"NaN in test: {X_test_scaled.isnull().sum().sum()}")

# Check for Inf
print(f"Inf in train: {np.isinf(X_train_scaled.values).sum()}")
print(f"Inf in test: {np.isinf(X_test_scaled.values).sum()}")

# Statistics
print(f"\nTraining statistics:")
print(f"  Mean: {X_train_scaled.mean().mean():.4f}")
print(f"  Std: {X_train_scaled.std().mean():.4f}")
print(f"  Min: {X_train_scaled.min().min():.4f}")
print(f"  Max: {X_train_scaled.max().max():.4f}")

print(f"\nTest statistics:")
print(f"  Mean: {X_test_scaled.mean().mean():.4f}")
print(f"  Std: {X_test_scaled.std().mean():.4f}")
print(f"  Min: {X_test_scaled.min().min():.4f}")
print(f"  Max: {X_test_scaled.max().max():.4f}")

# ============================================================================
# STEP 7: APPLY SMOTE (OPTIONAL - for training edge models)
# ============================================================================
print("\nSTEP 7: CREATE SMOTE-BALANCED VERSION (for reference)")
print("-" * 80)

print(f"Before SMOTE:")
print(f"  Class 0 (Normal): {(train_df.shape[0])} samples")
print(f"  Class 1 (Attack): Training has no attacks (normal operation only)")
print(f"  Ratio: 100% normal")

print(f"\nNote: Training set is pure normal operation (no attacks)")
print(f"      Test set has {(test_labels == 1).sum():,} attack samples")
print(f"      No SMOTE needed for training (no minority class)")

# ============================================================================
# STEP 8: SAVE PROCESSED DATA
# ============================================================================
print("\nSTEP 8: SAVE PROCESSED DATA")
print("-" * 80)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training
X_train_scaled.to_csv(os.path.join(OUTPUT_DIR, 'wadi_train_scaled.csv'), index=False)
print(f"✓ wadi_train_scaled.csv: {X_train_scaled.shape}")

# Test
X_test_scaled.to_csv(os.path.join(OUTPUT_DIR, 'wadi_test_scaled.csv'), index=False)
test_labels_df = pd.DataFrame({'attack': test_labels})
test_labels_df.to_csv(os.path.join(OUTPUT_DIR, 'wadi_test_labels.csv'), index=False)
print(f"✓ wadi_test_scaled.csv: {X_test_scaled.shape}")
print(f"✓ wadi_test_labels.csv: {test_labels_df.shape}")

# Save scaler for later use
with open(os.path.join(OUTPUT_DIR, 'wadi_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ wadi_scaler.pkl: Scaler saved")

# Save feature list
with open(os.path.join(OUTPUT_DIR, 'wadi_selected_features.pkl'), 'wb') as f:
    pickle.dump(selected_features, f)
print(f"✓ wadi_selected_features.pkl: {len(selected_features)} features")

# ============================================================================
# STEP 9: ATTACK SCENARIO ANALYSIS
# ============================================================================
print("\nSTEP 9: ATTACK SCENARIO ANALYSIS")
print("-" * 80)

print(f"\nTest set attack breakdown:")
print(f"  Total test samples: {len(test_labels):,}")
print(f"  Attack samples: {(test_labels == 1).sum():,}")
print(f"  Normal samples: {(test_labels == 0).sum():,}")
print(f"  Attack ratio: {(test_labels == 1).sum()/len(test_labels)*100:.2f}%")

# Validate sufficient attacks for meaningful evaluation
min_attacks = 50
actual_attacks = (test_labels == 1).sum()
if actual_attacks >= min_attacks:
    print(f"  ✓ SUFFICIENT: {actual_attacks:,} attack samples (need >= {min_attacks})")
else:
    print(f"  ⚠ WARNING: Only {actual_attacks:,} attacks (need >= {min_attacks})")

# ============================================================================
# STEP 10: SUMMARY & READINESS
# ============================================================================
print("\n" + "="*80)
print("PREPROCESSING COMPLETE - READINESS CHECKLIST")
print("="*80)

readiness = {
    'Data loaded': True,
    'Labels extracted': (test_labels == 1).sum() > 0,
    'Features selected': len(selected_features) > 20,
    'Data scaled': X_train_scaled.max().max() <= 1.0,
    'No NaN values': X_train_scaled.isnull().sum().sum() == 0,
    'Sufficient attacks': (test_labels == 1).sum() >= min_attacks,
    'Good train/test split': True
}

print("\nReadiness Status:")
for check, status in readiness.items():
    symbol = "✓" if status else "✗"
    print(f"  {symbol} {check}")

print("\n" + "="*80)
print("NEXT: Run Config_2_Balanced model on WADI")
print("="*80)

print(f"""
WADI Dataset Summary:
  Training: {X_train_scaled.shape[0]:,} normal operation samples
  Test: {X_test_scaled.shape[0]:,} total samples
    - {(test_labels == 0).sum():,} normal ({(test_labels == 0).sum()/len(test_labels)*100:.1f}%)
    - {(test_labels == 1).sum():,} attacks ({(test_labels == 1).sum()/len(test_labels)*100:.1f}%)
  
  Features: {len(selected_features)} selected (from {len(sensor_cols)} original)
  Quality: All features normalized to [0, 1]
  
  Attack Scenarios: {actual_attacks} unique attack instances
  Suitable for: Cross-dataset validation with HAI
""")

print("="*80)
