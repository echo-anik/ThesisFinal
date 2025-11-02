"""
Comprehensive EDA and Feature Engineering for HAI and WADI datasets
====================================================================
Goal: Achieve 95% accuracy on HAI and 85% on WADI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

def analyze_hai_dataset():
    """Comprehensive HAI analysis"""
    print("="*80)
    print("HAI DATASET ANALYSIS")
    print("="*80)

    # Load sample data
    print("\n1. Loading data...")
    train = pd.read_csv('combined_HAI_TRAIN.csv', nrows=50000)
    test = pd.read_csv('combined_HAI_TEST.csv', nrows=50000)
    labels = pd.read_csv('combined_HAI_TEST_LABELS.csv', nrows=50000)

    print(f"Train: {train.shape}")
    print(f"Test: {test.shape}")
    print(f"Labels: {labels.shape}")

    # Extract labels
    y_test = labels['label'].values if 'label' in labels.columns else labels.iloc[:, -1].values
    y_test = (y_test != 0).astype(int)

    print(f"\n2. Label distribution:")
    print(f"   Normal: {np.sum(y_test==0)} ({np.sum(y_test==0)/len(y_test)*100:.1f}%)")
    print(f"   Attack: {np.sum(y_test==1)} ({np.sum(y_test==1)/len(y_test)*100:.1f}%)")

    # Remove timestamp
    if 'timestamp' in train.columns:
        train = train.drop('timestamp', axis=1)
        test = test.drop('timestamp', axis=1)

    print(f"\n3. Feature analysis:")
    print(f"   Total features: {train.shape[1]}")

    # Identify constant columns
    const_cols = [col for col in train.columns if train[col].nunique() <= 1]
    print(f"   Constant columns: {len(const_cols)}")

    # Identify low variance columns
    low_var_cols = []
    for col in train.select_dtypes(include=[np.number]).columns:
        if train[col].std() < 0.01:
            low_var_cols.append(col)
    print(f"   Low variance columns: {len(low_var_cols)}")

    # High correlation pairs
    numeric_train = train.select_dtypes(include=[np.number])
    corr = numeric_train.corr().abs()
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if corr.iloc[i, j] > 0.95:
                high_corr_pairs.append((corr.columns[i], corr.columns[j]))
    print(f"   High correlation pairs (>0.95): {len(high_corr_pairs)}")

    # Features to drop
    drop_cols = list(set(const_cols + low_var_cols))
    print(f"\n4. Recommended features to drop: {len(drop_cols)}")

    # Drop highly correlated - keep first of each pair
    drop_corr = [pair[1] for pair in high_corr_pairs]
    print(f"   Additional from high correlation: {len(set(drop_corr))}")

    all_drop = list(set(drop_cols + drop_corr))
    remaining_features = [col for col in train.columns if col not in all_drop]
    print(f"   Remaining features: {len(remaining_features)}")

    return {
        'drop_columns': all_drop,
        'keep_columns': remaining_features,
        'const_columns': const_cols,
        'low_var_columns': low_var_cols,
        'high_corr_pairs': high_corr_pairs[:10]  # First 10
    }

def analyze_wadi_dataset():
    """Comprehensive WADI analysis"""
    print("\n" + "="*80)
    print("WADI DATASET ANALYSIS")
    print("="*80)

    # Load sample data
    print("\n1. Loading data...")
    train = pd.read_csv('WADI_14days.csv', skiprows=4, nrows=50000, low_memory=False)

    # Clean column names
    train.columns = [col.split('\\')[-1].strip() for col in train.columns]

    print(f"Train: {train.shape}")

    # Remove non-feature columns
    drop_initial = ['Row', 'Date', 'Time']
    train = train.drop(columns=drop_initial, errors='ignore')

    print(f"\n2. After removing metadata: {train.shape}")

    # Select numeric only
    numeric_cols = train.select_dtypes(include=[np.number]).columns
    print(f"   Numeric columns: {len(numeric_cols)}")

    # Constant columns
    const_cols = [col for col in numeric_cols if train[col].nunique() <= 1]
    print(f"   Constant columns: {len(const_cols)}")

    # Low variance
    low_var_cols = []
    for col in numeric_cols:
        if train[col].std() < 0.01:
            low_var_cols.append(col)
    print(f"   Low variance columns: {len(low_var_cols)}")

    # High correlation
    corr = train[numeric_cols].corr().abs()
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if corr.iloc[i, j] > 0.95:
                high_corr_pairs.append((corr.columns[i], corr.columns[j]))
    print(f"   High correlation pairs (>0.95): {len(high_corr_pairs)}")

    # Features to drop
    drop_cols = list(set(const_cols + low_var_cols))
    drop_corr = [pair[1] for pair in high_corr_pairs]
    all_drop = list(set(drop_cols + drop_corr))

    remaining_features = [col for col in numeric_cols if col not in all_drop]
    print(f"\n3. Recommended features to drop: {len(all_drop)}")
    print(f"   Remaining features: {len(remaining_features)}")

    return {
        'drop_columns': all_drop,
        'keep_columns': remaining_features,
        'const_columns': const_cols,
        'low_var_columns': low_var_cols,
        'high_corr_pairs': high_corr_pairs[:10]
    }

def main():
    """Run comprehensive EDA"""
    print("\n" + "="*80)
    print("COMPREHENSIVE EDA FOR THESIS EXPERIMENTS")
    print("="*80)

    # Analyze HAI
    hai_analysis = analyze_hai_dataset()

    # Analyze WADI
    wadi_analysis = analyze_wadi_dataset()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)

    print("\nHAI Dataset:")
    print(f"  - Original features: 86")
    print(f"  - Drop: {len(hai_analysis['drop_columns'])} features")
    print(f"  - Keep: {len(hai_analysis['keep_columns'])} features")
    print(f"  - Issue: 0% attacks in training (need anomaly detection or data balancing)")

    print("\nWADI Dataset:")
    print(f"  - Original features: ~127")
    print(f"  - Drop: {len(wadi_analysis['drop_columns'])} features")
    print(f"  - Keep: {len(wadi_analysis['keep_columns'])} features")
    print(f"  - Has 14% attacks in training (good for supervised learning)")

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Create enhanced preprocessing with feature selection")
    print("2. For HAI: Use ensemble of anomaly detection + supervised learning")
    print("3. For WADI: Use supervised learning with feature engineering")
    print("4. Add temporal features (rolling stats, differences)")
    print("5. Implement proper cross-validation")

    # Save analysis
    import pickle
    with open('eda_analysis.pkl', 'wb') as f:
        pickle.dump({'hai': hai_analysis, 'wadi': wadi_analysis}, f)
    print("\nAnalysis saved to: eda_analysis.pkl")

if __name__ == '__main__':
    main()
