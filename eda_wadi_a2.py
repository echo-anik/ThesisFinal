"""
WADI A2 DATASET EXPLORATORY DATA ANALYSIS
==========================================
Comprehensive EDA for WADI A2 (19 Nov 2019) dataset
Analyzes training (14 days normal) and test (attack data with labels)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Configuration
WADI_DIR = r'g:\THESIS\WADI.A2_19 Nov 2019'
TRAIN_FILE = os.path.join(WADI_DIR, 'WADI_14days_new.csv')
TEST_FILE = os.path.join(WADI_DIR, 'WADI_attackdataLABLE.csv')

print("="*80)
print("WADI A2 DATASET EXPLORATORY DATA ANALYSIS")
print("="*80)
print()

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("STEP 1: LOADING DATA")
print("-" * 80)

print(f"Loading training data from: {TRAIN_FILE}")
train_df = pd.read_csv(TRAIN_FILE)
print(f"✓ Training data loaded: {train_df.shape[0]} rows × {train_df.shape[1]} columns")

print(f"\nLoading test data from: {TEST_FILE}")
test_df = pd.read_csv(TEST_FILE)
print(f"✓ Test data loaded: {test_df.shape[0]} rows × {test_df.shape[1]} columns")

# ============================================================================
# STEP 2: BASIC STRUCTURE
# ============================================================================
print("\n" + "="*80)
print("STEP 2: BASIC DATA STRUCTURE")
print("-" * 80)

print("\nTRAINING DATA:")
print(f"  Rows: {train_df.shape[0]:,}")
print(f"  Columns: {train_df.shape[1]}")
print(f"  Memory: {train_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nTEST DATA:")
print(f"  Rows: {test_df.shape[0]:,}")
print(f"  Columns: {test_df.shape[1]}")
print(f"  Memory: {test_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# STEP 3: COLUMN ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 3: COLUMN NAMES & TYPES")
print("-" * 80)

print("\nTRAINING COLUMNS (first 20):")
for i, col in enumerate(train_df.columns[:20]):
    dtype = train_df[col].dtype
    print(f"  {i}: {col} ({dtype})")

print(f"\n... and {train_df.shape[1]-20} more columns")

print("\nTEST COLUMNS (first 20):")
for i, col in enumerate(test_df.columns[:20]):
    dtype = test_df[col].dtype
    print(f"  {i}: {col} ({dtype})")

print(f"\n... and {test_df.shape[1]-20} more columns")

# Check for label column
if 'attack' in test_df.columns:
    print("\n✓ FOUND: 'attack' label column in test data")
    print(f"  Attack values: {test_df['attack'].unique()}")
    print(f"  Attack distribution:\n{test_df['attack'].value_counts()}")
elif 'Attack' in test_df.columns:
    print("\n✓ FOUND: 'Attack' label column in test data")
    print(f"  Attack values: {test_df['Attack'].unique()}")
    print(f"  Attack distribution:\n{test_df['Attack'].value_counts()}")
else:
    print("\n⚠ WARNING: No 'attack' or 'Attack' column found")
    print("Available columns (last 10):")
    for col in test_df.columns[-10:]:
        print(f"  - {col}")

# ============================================================================
# STEP 4: DATA TYPES & MISSING VALUES
# ============================================================================
print("\n" + "="*80)
print("STEP 4: DATA TYPES & MISSING VALUES")
print("-" * 80)

print("\nTRAINING DATA:")
print(f"  Data types:\n{train_df.dtypes.value_counts()}")
print(f"\n  Missing values:\n{train_df.isnull().sum().sum()} total")
print(f"  Missing by column: {train_df.isnull().any().sum()} columns have nulls")

print("\nTEST DATA:")
print(f"  Data types:\n{test_df.dtypes.value_counts()}")
print(f"\n  Missing values:\n{test_df.isnull().sum().sum()} total")
print(f"  Missing by column: {test_df.isnull().any().sum()} columns have nulls")

# ============================================================================
# STEP 5: HEAD & TAIL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("STEP 5: FIRST & LAST ROWS")
print("-" * 80)

print("\nTRAINING - First 3 rows:")
print(train_df.head(3).to_string())

print("\n\nTRAINING - Last 3 rows:")
print(train_df.tail(3).to_string())

print("\n\nTEST - First 3 rows:")
print(test_df.head(3).to_string())

print("\n\nTEST - Last 3 rows:")
print(test_df.tail(3).to_string())

# ============================================================================
# STEP 6: FEATURE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("STEP 6: STATISTICAL SUMMARY")
print("-" * 80)

print("\nTRAINING DATA - Numeric columns:")
numeric_train = train_df.select_dtypes(include=[np.number])
print(f"  Numeric columns: {numeric_train.shape[1]}")
print(f"\n{numeric_train.describe().to_string()}")

print("\n\nTEST DATA - Numeric columns:")
numeric_test = test_df.select_dtypes(include=[np.number])
print(f"  Numeric columns: {numeric_test.shape[1]}")
print(f"\n{numeric_test.describe().to_string()}")

# ============================================================================
# STEP 7: DUPLICATES & UNIQUE VALUES
# ============================================================================
print("\n" + "="*80)
print("STEP 7: DUPLICATES & UNIQUE VALUES")
print("-" * 80)

print("\nTRAINING DATA:")
print(f"  Duplicate rows: {train_df.duplicated().sum()}")
print(f"  Unique combinations: {len(train_df.drop_duplicates())}")

print("\nTEST DATA:")
print(f"  Duplicate rows: {test_df.duplicated().sum()}")
print(f"  Unique combinations: {len(test_df.drop_duplicates())}")

# ============================================================================
# STEP 8: FEATURE COMPARISON (Train vs Test)
# ============================================================================
print("\n" + "="*80)
print("STEP 8: TRAIN vs TEST FEATURE COMPARISON")
print("-" * 80)

# Get common numeric columns
train_numeric_cols = numeric_train.columns.tolist()
test_numeric_cols = numeric_test.columns.tolist()
common_cols = [c for c in train_numeric_cols if c in test_numeric_cols]

print(f"\nCommon numeric columns: {len(common_cols)}")
print(f"Train-only columns: {len([c for c in train_numeric_cols if c not in test_numeric_cols])}")
print(f"Test-only columns: {len([c for c in test_numeric_cols if c not in train_numeric_cols])}")

if common_cols:
    print(f"\nComparison of first 5 common features:")
    for col in common_cols[:5]:
        train_mean = train_df[col].mean() if col in train_df else np.nan
        train_std = train_df[col].std() if col in train_df else np.nan
        test_mean = test_df[col].mean() if col in test_df else np.nan
        test_std = test_df[col].std() if col in test_df else np.nan
        
        print(f"\n  {col}:")
        print(f"    Train: mean={train_mean:.4f}, std={train_std:.4f}")
        print(f"    Test:  mean={test_mean:.4f}, std={test_std:.4f}")
        print(f"    Drift: {abs(train_mean - test_mean):.4f}")

# ============================================================================
# STEP 9: COLUMN NAME PATTERNS
# ============================================================================
print("\n" + "="*80)
print("STEP 9: COLUMN NAME PATTERNS")
print("-" * 80)

print("\nTRAINING - Unique prefixes (first 5 chars):")
prefixes = {}
for col in train_df.columns:
    prefix = col[:5] if len(col) > 5 else col[:3]
    prefixes[prefix] = prefixes.get(prefix, 0) + 1

for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1])[:15]:
    print(f"  {prefix}: {count} columns")

# ============================================================================
# STEP 10: SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 10: SUMMARY & RECOMMENDATIONS")
print("-" * 80)

print("\nDATASET CHARACTERISTICS:")
print(f"  ✓ Training set: {train_df.shape[0]:,} normal operation samples")
print(f"  ✓ Test set: {test_df.shape[0]:,} samples with {len(numeric_test.columns)} features")
print(f"  ✓ Time period: 14 days normal + attack scenarios")

# Check label distribution in test
if 'attack' in test_df.columns:
    attack_ratio = (test_df['attack'] == 1).sum() / len(test_df) * 100
    print(f"  ✓ Attack rate in test: {attack_ratio:.2f}%")
elif 'Attack' in test_df.columns:
    attack_ratio = (test_df['Attack'] == 1).sum() / len(test_df) * 100
    print(f"  ✓ Attack rate in test: {attack_ratio:.2f}%")

print("\nRECOMMENDED NEXT STEPS:")
print("  1. ✓ Align column names between train and test")
print("  2. ✓ Handle missing values (if any)")
print("  3. ✓ Scale/normalize all features")
print("  4. ✓ Feature selection (remove low-variance features)")
print("  5. ✓ Apply SMOTE for class balance in training")
print("  6. ✓ Train hierarchical models (RF edge + GB central)")
print("  7. ✓ Evaluate on test with proper labels")

print("\n" + "="*80)
print("EDA COMPLETE")
print("="*80)
