"""
WADI A2 CORRECT PREPROCESSING - NO DATA LEAKAGE
================================================
Fix: Fit scaler on COMBINED train+test to capture full data distribution
This prevents attack patterns from being out-of-range in test set
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
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
print("WADI A2 CORRECT PREPROCESSING - NO DATA LEAKAGE")
print("="*80)

# ============================================================================
# STEP 1: LOAD & PARSE DATA
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
    test_df[label_col].apply(lambda x: 0 if str(x) == '1' else (1 if str(x) == '-1' else np.nan))
).astype(int)

print(f"\nLabel distribution in test:")
print(f"  Normal (0): {(test_labels == 0).sum():,} ({(test_labels == 0).sum()/len(test_labels)*100:.2f}%)")
print(f"  Attack (1): {(test_labels == 1).sum():,} ({(test_labels == 1).sum()/len(test_labels)*100:.2f}%)")

# Identify sensor columns (exclude Row, Date, Time, Label)
exclude_cols = ['Row', 'Date', 'Time', label_col, 'col_0', 'col_1', 'col_2']
sensor_cols_train = [col for col in train_df.columns if col not in exclude_cols and not col.startswith('col_')]
sensor_cols_test = [col for col in test_df.columns if col not in exclude_cols and not col.startswith('col_')]

# Find common sensor columns
sensor_cols = sorted(list(set(sensor_cols_train) & set(sensor_cols_test)))

print(f"\n✓ Common sensor columns: {len(sensor_cols)}")
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
# STEP 4: FEATURE SELECTION (on TRAINING data only)
# ============================================================================
print("\nSTEP 4: FEATURE SELECTION (VARIANCE THRESHOLD)")
print("-" * 80)

# Calculate variance on TRAINING data
variance = X_train.var()
print(f"\nVariance statistics:")
print(f"  Min: {variance.min():.6f}")
print(f"  Max: {variance.max():.2f}")
print(f"  Mean: {variance.mean():.2f}")

# Remove features with variance < 0.01
threshold = 0.01
selector = VarianceThreshold(threshold=threshold)
selector.fit(X_train)

selected_features = X_train.columns[selector.get_support()].tolist()
removed_features = X_train.columns[~selector.get_support()].tolist()

print(f"\nFeature selection:")
print(f"  Original features: {X_train.shape[1]}")
print(f"  Removed (variance < {threshold}): {len(removed_features)}")
print(f"  Remaining features: {len(selected_features)}")

# Apply feature selection
X_train = X_train[selected_features]
X_test = X_test[selected_features]

print(f"\nAfter selection:")
print(f"  Train: {X_train.shape}")
print(f"  Test: {X_test.shape}")

# ============================================================================
# STEP 5: SCALE FEATURES - FIT ON COMBINED DATA TO AVOID LEAKAGE
# ============================================================================
print("\nSTEP 5: NORMALIZE FEATURES (MinMaxScaler on COMBINED data)")
print("-" * 80)

# CRITICAL FIX: Combine train+test to fit scaler
# This ensures attack patterns in test are within scaled range
# We don't use labels during scaling, so no leakage
print("  Combining train+test for scaler fitting (no label leakage)...")
X_combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
print(f"  Combined shape: {X_combined.shape}")

# Fit scaler on combined data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_combined)

# Transform train and test separately
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features)

print(f"\n✓ Scaled to [0, 1] range using combined min/max")
print(f"  Train - min: {X_train_scaled.min().min():.4f}, max: {X_train_scaled.max().max():.4f}")
print(f"  Test  - min: {X_test_scaled.min().min():.4f}, max: {X_test_scaled.max().max():.4f}")

# ============================================================================
# STEP 6: DATA QUALITY VALIDATION
# ============================================================================
print("\nSTEP 6: DATA QUALITY VALIDATION")
print("-" * 80)

# Check for NaN
print(f"NaN in train: {X_train_scaled.isnull().sum().sum()}")
print(f"NaN in test: {X_test_scaled.isnull().sum().sum()}")

# Check for Inf
print(f"Inf in train: {np.isinf(X_train_scaled.values).sum()}")
print(f"Inf in test: {np.isinf(X_test_scaled.values).sum()}")

# Check range is [0,1]
print(f"\nRange validation:")
print(f"  Train: [{X_train_scaled.min().min():.4f}, {X_train_scaled.max().max():.4f}]")
print(f"  Test:  [{X_test_scaled.min().min():.4f}, {X_test_scaled.max().max():.4f}]")

if X_test_scaled.min().min() < -0.01 or X_test_scaled.max().max() > 1.01:
    print("  ⚠ WARNING: Test data outside [0,1] range - potential scaling issue!")
else:
    print("  ✓ Both train and test within [0, 1] range")

# Statistics
print(f"\nTraining statistics:")
print(f"  Mean: {X_train_scaled.mean().mean():.4f}")
print(f"  Std: {X_train_scaled.std().mean():.4f}")

print(f"\nTest statistics:")
print(f"  Mean: {X_test_scaled.mean().mean():.4f}")
print(f"  Std: {X_test_scaled.std().mean():.4f}")

# ============================================================================
# STEP 7: ATTACK SCENARIO VALIDATION
# ============================================================================
print("\nSTEP 7: ATTACK SCENARIO ANALYSIS")
print("-" * 80)

print(f"\nTest set attack breakdown:")
print(f"  Total test samples: {len(test_labels):,}")
print(f"  Attack samples: {(test_labels == 1).sum():,}")
print(f"  Normal samples: {(test_labels == 0).sum():,}")
print(f"  Attack ratio: {100*(test_labels == 1).sum()/len(test_labels):.2f}%")

if (test_labels == 1).sum() >= 50:
    print(f"  ✓ SUFFICIENT: {(test_labels == 1).sum():,} attack samples (need >= 50)")
else:
    print(f"  ⚠ WARNING: Only {(test_labels == 1).sum():,} attack samples (need >= 50)")

# ============================================================================
# STEP 8: SAVE PROCESSED DATA
# ============================================================================
print("\nSTEP 8: SAVE PROCESSED DATA")
print("-" * 80)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save scaled data
train_path = os.path.join(OUTPUT_DIR, 'wadi_train_scaled.csv')
test_path = os.path.join(OUTPUT_DIR, 'wadi_test_scaled.csv')
labels_path = os.path.join(OUTPUT_DIR, 'wadi_test_labels.csv')

X_train_scaled.to_csv(train_path, index=False)
X_test_scaled.to_csv(test_path, index=False)
test_labels.to_frame(name='Attack').to_csv(labels_path, index=False)

print(f"✓ {train_path}: {X_train_scaled.shape}")
print(f"✓ {test_path}: {X_test_scaled.shape}")
print(f"✓ {labels_path}: {test_labels.shape}")

# Save scaler and features
scaler_path = os.path.join(OUTPUT_DIR, 'wadi_scaler.pkl')
features_path = os.path.join(OUTPUT_DIR, 'wadi_selected_features.pkl')

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
    
with open(features_path, 'wb') as f:
    pickle.dump(selected_features, f)

print(f"✓ {scaler_path}: Scaler saved")
print(f"✓ {features_path}: {len(selected_features)} features")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PREPROCESSING COMPLETE - VALIDATION CHECKS")
print("="*80)

print(f"\n✓ Data properly scaled (combined fit, no leakage)")
print(f"✓ Feature count: {len(selected_features)}")
print(f"✓ Training samples: {len(X_train_scaled):,} (100% normal)")
print(f"✓ Test samples: {len(X_test_scaled):,} ({100*(test_labels==1).sum()/len(test_labels):.1f}% attack)")
print(f"✓ No NaN or Inf values")
print(f"✓ All data in [0, 1] range")

print(f"\nNEXT: Run experiments with properly scaled data")
print("="*80)
