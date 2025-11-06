"""
WADI A2 DATASET - QUICK ANALYSIS
=================================
Identifies structure and issues
"""

import pandas as pd
import numpy as np
import os

# Use relative paths from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WADI_DIR = os.path.join(SCRIPT_DIR, 'WADI.A2_19 Nov 2019')
TRAIN_FILE = os.path.join(WADI_DIR, 'WADI_14days_new.csv')
TEST_FILE = os.path.join(WADI_DIR, 'WADI_attackdataLABLE.csv')

print("\n" + "="*80)
print("WADI A2 DATASET - STRUCTURE ANALYSIS")
print("="*80)

# Load
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"\n✓ Training: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
print(f"✓ Test: {test_df.shape[0]:,} rows × {test_df.shape[1]} columns")

# ISSUE DETECTION
print("\n" + "="*80)
print("CRITICAL ISSUES")
print("="*80)

print(f"\nTraining Dtypes:")
print(train_df.dtypes.value_counts())

print(f"\nTest Dtypes:")
print(test_df.dtypes.value_counts())

print("\n⚠ ISSUE: Test data has all OBJECT (text) type!")
print("Likely cause: CSV parsing issue with special characters or formatting")

# CHECK FIRST TEST ROW
print("\n" + "="*80)
print("TEST FILE - First Row Inspection")
print("="*80)
print("\nFirst row column names:")
print(test_df.columns[:10].tolist())

print("\nFirst row values (first 5 columns):")
for i, col in enumerate(test_df.columns[:5]):
    print(f"  {col}: {test_df[col].iloc[0]}")

# Try to identify the label column
print("\n" + "="*80)
print("LABEL COLUMN SEARCH")
print("="*80)

# Check last columns (likely to have labels)
print("\nLast 5 columns:")
for col in test_df.columns[-5:]:
    print(f"  {col}")
    unique_vals = test_df[col].unique()[:5]
    print(f"    Sample values: {unique_vals}")

# Check for 'attack' or 'Attack' or 'LABLE' or similar
label_candidates = [col for col in test_df.columns if 'attack' in col.lower() or 'label' in col.lower() or 'lable' in col.lower()]
print(f"\nLabel column candidates: {label_candidates}")

if label_candidates:
    for col in label_candidates:
        print(f"\n  Column: {col}")
        print(f"    Unique values: {test_df[col].unique()}")
        print(f"    Value counts:\n{test_df[col].value_counts()}")

# SOLUTION
print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
The test file appears to have parsing issues. Need to:

1. ✓ Open test CSV manually to check separator (comma, tab, semicolon?)
2. ✓ Verify column count matches (should be 131)
3. ✓ Check for encoding issues (UTF-8, Latin-1, etc.)
4. ✓ Look for the label column (Attack, LABLE, label, etc.)
5. ✓ Re-parse with correct separator and encoding

Recommendation: Save test file in clean format and reload
""")

print("\n" + "="*80)
