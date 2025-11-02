"""
WADI A2 PREPROCESSING - PROPER LOADING & EDA
=============================================
Fixes header parsing and prepares data for modeling
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

WADI_DIR = r'g:\THESIS\WADI.A2_19 Nov 2019'
TRAIN_FILE = os.path.join(WADI_DIR, 'WADI_14days_new.csv')
TEST_FILE = os.path.join(WADI_DIR, 'WADI_attackdataLABLE.csv')
OUTPUT_DIR = os.path.join(r'g:\THESIS\processed_data')

print("="*80)
print("WADI A2 PREPROCESSING & EDA")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA WITH CORRECT HEADERS
# ============================================================================
print("\nSTEP 1: LOADING DATA")
print("-" * 80)

# Load training
print("Loading training data...")
train_df = pd.read_csv(TRAIN_FILE, low_memory=False)

# Load test - skip first row that contains headers in data
print("Loading test data...")
test_df = pd.read_csv(TEST_FILE, low_memory=False)

# The test file has headers in first data row
# Column indices in raw data are: 0=Row, 1=Date, 2=Time, 3-129=sensors, 130=Label
# Extract header row
header_row = test_df.iloc[0].values
actual_headers = [f"col_{i}" if isinstance(v, (int, float)) else str(v) for i, v in enumerate(header_row)]

# Remove first row and set proper headers
test_df = test_df.iloc[1:].reset_index(drop=True)
test_df.columns = actual_headers

print(f"✓ Training: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
print(f"✓ Test: {test_df.shape[0]:,} rows × {test_df.shape[1]} columns")

# ============================================================================
# STEP 2: FIND AND EXTRACT LABEL COLUMN
# ============================================================================
print("\nSTEP 2: EXTRACT LABEL COLUMN")
print("-" * 80)

# Look for label column in last columns
label_col = None
label_mapping = {}

print("Last 3 columns:")
for col in test_df.columns[-3:]:
    print(f"  {col}: {test_df[col].unique()[:5].tolist()}")
    if 'Attack' in str(col) or 'LABLE' in str(col) or 'attack' in str(col).lower():
        label_col = col
        print(f"    → FOUND LABEL COLUMN!")

if label_col:
    print(f"\n✓ Label column: {label_col}")
    unique_labels = test_df[label_col].unique()
    print(f"  Unique values: {unique_labels}")
    
    # Create label mapping: 1 = No Attack (normal=0), -1 = Attack (attack=1)
    label_mapping = {1: 0, -1: 1}  # -1 = attack, 1 = normal
    print(f"  Mapping: {label_mapping}")
    
    test_labels = test_df[label_col].map(label_mapping)
    print(f"  Attack distribution: {test_labels.value_counts().to_dict()}")
else:
    print("\n⚠ WARNING: Label column not found!")

# ============================================================================
# STEP 3: SELECT SENSOR COLUMNS
# ============================================================================
print("\nSTEP 3: IDENTIFY SENSOR COLUMNS")
print("-" * 80)

# Get training headers
train_headers = train_df.columns.tolist()
print(f"Training columns: {len(train_headers)}")
print(f"  First: {train_headers[:3]}")
print(f"  Last: {train_headers[-3:]}")

# Sensor columns are those with IDs like 1_*, 2_*, 3_*
sensor_cols = [col for col in train_headers if any(col.startswith(prefix) for prefix in ['1_', '2_', '3_', '2A_', '2B_'])]
print(f"\n✓ Found {len(sensor_cols)} sensor columns")

# For test data, match sensor names from training
test_sensor_cols = []
for col in sensor_cols:
    # Find matching column in test
    if col in test_df.columns:
        test_sensor_cols.append(col)

print(f"✓ Matching sensor columns in test: {len(test_sensor_cols)}")

# ============================================================================
# STEP 4: CLEAN & CONVERT DATA
# ============================================================================
print("\nSTEP 4: CLEAN & CONVERT DATA TYPES")
print("-" * 80)

# Training
X_train = train_df[sensor_cols].copy()
print(f"Training X shape: {X_train.shape}")

# Test
X_test = test_df[test_sensor_cols].copy() if test_sensor_cols else test_df[sensor_cols].copy()
print(f"Test X shape: {X_test.shape}")
print(f"Test y shape: {test_labels.shape}")

# Convert to numeric (force non-numeric to NaN)
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')
test_labels = pd.to_numeric(test_labels, errors='coerce')

print(f"\nMissing values - Train: {X_train.isnull().sum().sum()}")
print(f"Missing values - Test: {X_test.isnull().sum().sum()}")

# Handle missing values - forward fill then backward fill
X_train = X_train.fillna(method='ffill').fillna(method='bfill')
X_test = X_test.fillna(method='ffill').fillna(method='bfill')
test_labels = test_labels.dropna()

print(f"After fill - Train: {X_train.isnull().sum().sum()}")
print(f"After fill - Test: {X_test.isnull().sum().sum()}")

# ============================================================================
# STEP 5: STATISTICS
# ============================================================================
print("\nSTEP 5: DATA STATISTICS")
print("-" * 80)

print(f"\nTraining X:")
print(f"  Shape: {X_train.shape}")
print(f"  Min/Max: {X_train.min().min():.2f} / {X_train.max().max():.2f}")
print(f"  Mean/Std: {X_train.mean().mean():.2f} / {X_train.std().mean():.2f}")

print(f"\nTest X:")
print(f"  Shape: {X_test.shape}")
print(f"  Min/Max: {X_test.min().min():.2f} / {X_test.max().max():.2f}")
print(f"  Mean/Std: {X_test.mean().mean():.2f} / {X_test.std().mean():.2f}")

print(f"\nTest y:")
print(f"  Normal (0): {(test_labels == 0).sum()} ({(test_labels == 0).sum()/len(test_labels)*100:.2f}%)")
print(f"  Attack (1): {(test_labels == 1).sum()} ({(test_labels == 1).sum()/len(test_labels)*100:.2f}%)")

# ============================================================================
# STEP 6: SAVE PROCESSED DATA
# ============================================================================
print("\nSTEP 6: SAVING PROCESSED DATA")
print("-" * 80)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Train
X_train.to_csv(os.path.join(OUTPUT_DIR, 'wadi_train_raw.csv'), index=False)
print(f"✓ Saved: wadi_train_raw.csv ({X_train.shape[0]:,} × {X_train.shape[1]})")

# Test
X_test.to_csv(os.path.join(OUTPUT_DIR, 'wadi_test_raw.csv'), index=False)
test_labels.to_csv(os.path.join(OUTPUT_DIR, 'wadi_test_labels.csv'), index=False, header=['attack'])
print(f"✓ Saved: wadi_test_raw.csv ({X_test.shape[0]:,} × {X_test.shape[1]})")
print(f"✓ Saved: wadi_test_labels.csv")

# ============================================================================
# STEP 7: FEATURE ANALYSIS
# ============================================================================
print("\nSTEP 7: FEATURE ANALYSIS & SELECTION")
print("-" * 80)

# Remove constant/near-constant features
feature_variance = X_train.var()
constant_features = feature_variance[feature_variance < 0.001].index.tolist()
print(f"\nConstant features (<0.001 variance): {len(constant_features)}")
if constant_features:
    print(f"  Examples: {constant_features[:5]}")

# Remove highly correlated features
print(f"\nFeature range analysis (first 10):")
for i, col in enumerate(X_train.columns[:10]):
    col_min = X_train[col].min()
    col_max = X_train[col].max()
    col_range = col_max - col_min
    print(f"  {col}: [{col_min:.2f}, {col_max:.2f}] range={col_range:.2f}")

print("\n" + "="*80)
print("PREPROCESSING COMPLETE")
print("="*80)
print(f"""
Next steps:
1. Scale data with MinMaxScaler
2. Remove constant/low-variance features
3. Handle missing values properly
4. Apply SMOTE for class balance
5. Train Config_2_Balanced model (RF edge + GB central)
6. Evaluate on test set with labels
""")
