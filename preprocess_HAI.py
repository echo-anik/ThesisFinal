"""
HAI Dataset Preprocessing Pipeline
===================================
Processes the HAI (Hardware-in-the-loop Augmented ICS) dataset for hierarchical IDS experiments.
Handles large files efficiently for systems with limited RAM (16GB).

Features:
- Chunked processing for memory efficiency
- Checkpoint saving for power interruption recovery
- Progress tracking
- Data validation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import pickle
import gc
from datetime import datetime

class HAIPreprocessor:
    def __init__(self, base_dir=None, chunk_size=50000):
        # Use script directory if base_dir not provided
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir
        self.chunk_size = chunk_size
        self.processed_dir = os.path.join(base_dir, 'processed_data')
        self.checkpoint_dir = os.path.join(base_dir, 'Results', 'checkpoints')

        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.train_file = os.path.join(base_dir, 'combined_HAI_TRAIN.csv')
        self.test_file = os.path.join(base_dir, 'combined_HAI_TEST.csv')
        self.test_labels_file = os.path.join(base_dir, 'combined_HAI_TEST_LABELS.csv')

    def check_checkpoint(self, stage):
        """Check if a checkpoint exists for a processing stage"""
        checkpoint_file = os.path.join(self.checkpoint_dir, f'hai_{stage}_checkpoint.pkl')
        if os.path.exists(checkpoint_file):
            print(f"  OK - Found checkpoint for {stage}")
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return None

    def save_checkpoint(self, stage, data):
        """Save checkpoint for a processing stage"""
        checkpoint_file = os.path.join(self.checkpoint_dir, f'hai_{stage}_checkpoint.pkl')
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"  OK - Checkpoint saved for {stage}")

    def clear_checkpoint(self, stage):
        """Clear checkpoint after successful completion"""
        checkpoint_file = os.path.join(self.checkpoint_dir, f'hai_{stage}_checkpoint.pkl')
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

    def process_in_chunks(self, file_path, has_labels=False, label_file=None):
        """
        Process large CSV file in chunks to manage memory

        Args:
            file_path: Path to data file
            has_labels: Whether this is a labeled dataset
            label_file: Separate label file (for HAI test set)
        """
        print(f"\n  Processing {os.path.basename(file_path)} in chunks...")

        chunks = []
        total_rows = 0

        # First pass: determine columns
        first_chunk = pd.read_csv(file_path, nrows=1000)
        print(f"  Sample loaded: {first_chunk.shape}")

        # Identify columns to drop (non-numeric, identifiers)
        cols_to_drop = []
        for col in first_chunk.columns:
            if col.lower() in ['time', 'date', 'timestamp', 'row', 'index', 'attack']:
                cols_to_drop.append(col)

        print(f"  Dropping columns: {cols_to_drop}")

        # Process in chunks
        chunk_iter = pd.read_csv(file_path, chunksize=self.chunk_size)

        for i, chunk in enumerate(chunk_iter):
            # Drop non-feature columns
            chunk = chunk.drop(columns=cols_to_drop, errors='ignore')

            # Keep only numeric columns
            chunk = chunk.select_dtypes(include=[np.number])

            # Handle missing values
            chunk = chunk.fillna(chunk.mean())

            chunks.append(chunk)
            total_rows += len(chunk)

            if (i + 1) % 10 == 0:
                print(f"    Processed {total_rows:,} rows...")
                gc.collect()  # Force garbage collection

        print(f"  OK - Loaded {total_rows:,} rows total")

        # Concatenate all chunks
        print("  Concatenating chunks...")
        data = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()

        # Load labels if specified
        labels = None
        if label_file:
            print(f"  Loading labels from {os.path.basename(label_file)}...")
            labels_df = pd.read_csv(label_file)

            # Extract label column
            if 'label' in labels_df.columns:
                labels = labels_df['label'].values
            elif 'attack' in labels_df.columns:
                labels = labels_df['attack'].values
            elif 'Attack' in labels_df.columns:
                labels = labels_df['Attack'].values
            else:
                # Use last column if no label column found
                labels = labels_df.iloc[:, -1].values

            # Ensure binary labels (0 = normal, 1 = attack)
            labels = (labels != 0).astype(int)

            print(f"  OK - Labels loaded: {len(labels)} samples")
            print(f"    Attack ratio: {(labels == 1).sum() / len(labels) * 100:.2f}%")

        return data, labels

    def preprocess_hai_dataset(self, force_reprocess=False):
        """
        Main preprocessing pipeline for HAI dataset

        Args:
            force_reprocess: If True, ignore checkpoints and reprocess everything
        """
        print("="*80)
        print("HAI DATASET PREPROCESSING")
        print("="*80)
        print(f"Base directory: {self.base_dir}")
        print(f"Chunk size: {self.chunk_size:,} rows")

        # Check if already processed
        output_files = {
            'train_features': os.path.join(self.processed_dir, 'hai_train_scaled.csv'),
            'train_labels': os.path.join(self.processed_dir, 'hai_train_labels.csv'),
            'test_features': os.path.join(self.processed_dir, 'hai_test_scaled.csv'),
            'test_labels': os.path.join(self.processed_dir, 'hai_test_labels.csv'),
            'scaler': os.path.join(self.processed_dir, 'hai_scaler.pkl')
        }

        all_exist = all(os.path.exists(f) for f in output_files.values())

        if all_exist and not force_reprocess:
            print("\nOK - HAI dataset already processed!")
            print("  Use force_reprocess=True to reprocess.")
            return self.load_processed_data()

        # Step 1: Load and process training data
        print("\n" + "-"*80)
        print("STEP 1: Processing Training Data")
        print("-"*80)

        checkpoint = self.check_checkpoint('train_data')
        if checkpoint and not force_reprocess:
            X_train = checkpoint
        else:
            X_train, _ = self.process_in_chunks(self.train_file)
            self.save_checkpoint('train_data', X_train)

        print(f"  Training data shape: {X_train.shape}")

        # Create training labels (HAI train is all normal data = 0)
        y_train = np.zeros(len(X_train), dtype=int)
        print(f"  Training labels: {len(y_train)} samples (all normal)")

        # Step 2: Load and process test data
        print("\n" + "-"*80)
        print("STEP 2: Processing Test Data")
        print("-"*80)

        checkpoint = self.check_checkpoint('test_data')
        if checkpoint and not force_reprocess:
            X_test, y_test = checkpoint
        else:
            X_test, y_test = self.process_in_chunks(
                self.test_file,
                has_labels=True,
                label_file=self.test_labels_file
            )
            self.save_checkpoint('test_data', (X_test, y_test))

        print(f"  Test data shape: {X_test.shape}")
        print(f"  Test labels: {len(y_test)} samples")

        # Align columns between train and test
        print("\n" + "-"*80)
        print("STEP 3: Aligning Features")
        print("-"*80)

        common_cols = X_train.columns.intersection(X_test.columns)
        print(f"  Train features: {len(X_train.columns)}")
        print(f"  Test features: {len(X_test.columns)}")
        print(f"  Common features: {len(common_cols)}")

        X_train = X_train[common_cols]
        X_test = X_test[common_cols]

        # Step 4: Scale features
        print("\n" + "-"*80)
        print("STEP 4: Scaling Features")
        print("-"*80)

        checkpoint = self.check_checkpoint('scaler')
        if checkpoint and not force_reprocess:
            scaler = checkpoint
        else:
            print("  Fitting scaler on training data...")
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            self.save_checkpoint('scaler', scaler)

        print("  Transforming training data...")
        X_train_scaled = pd.DataFrame(
            scaler.transform(X_train),
            columns=X_train.columns
        )

        print("  Transforming test data...")
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns
        )

        # Step 5: Save processed data
        print("\n" + "-"*80)
        print("STEP 5: Saving Processed Data")
        print("-"*80)

        print("  Saving training features...")
        X_train_scaled.to_csv(output_files['train_features'], index=False)

        print("  Saving training labels...")
        pd.DataFrame({'attack': y_train}).to_csv(output_files['train_labels'], index=False)

        print("  Saving test features...")
        X_test_scaled.to_csv(output_files['test_features'], index=False)

        print("  Saving test labels...")
        pd.DataFrame({'attack': y_test}).to_csv(output_files['test_labels'], index=False)

        print("  Saving scaler...")
        with open(output_files['scaler'], 'wb') as f:
            pickle.dump(scaler, f)

        # Clear checkpoints
        for stage in ['train_data', 'test_data', 'scaler']:
            self.clear_checkpoint(stage)

        # Final summary
        print("\n" + "="*80)
        print("HAI PREPROCESSING COMPLETE")
        print("="*80)
        print(f"Training samples: {len(X_train_scaled):,}")
        print(f"Test samples: {len(X_test_scaled):,}")
        print(f"Features: {X_train_scaled.shape[1]}")
        print(f"Attack rate (train): {(y_train == 1).sum() / len(y_train) * 100:.2f}%")
        print(f"Attack rate (test): {(y_test == 1).sum() / len(y_test) * 100:.2f}%")
        print(f"\nFiles saved to: {self.processed_dir}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def load_processed_data(self):
        """Load already processed data"""
        print("\nLoading processed HAI data...")

        X_train = pd.read_csv(os.path.join(self.processed_dir, 'hai_train_scaled.csv'))
        X_test = pd.read_csv(os.path.join(self.processed_dir, 'hai_test_scaled.csv'))
        y_train = pd.read_csv(os.path.join(self.processed_dir, 'hai_train_labels.csv')).values.ravel()
        y_test = pd.read_csv(os.path.join(self.processed_dir, 'hai_test_labels.csv')).values.ravel()

        print(f"OK - Loaded: {len(X_train):,} train, {len(X_test):,} test samples")

        return X_train, X_test, y_train, y_test


def main():
    """Main execution"""
    print("\n" + "="*80)
    print(" HAI DATASET PREPROCESSING FOR HIERARCHICAL IDS")
    print("="*80)
    print(f" Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Initialize preprocessor
    preprocessor = HAIPreprocessor(chunk_size=50000)

    # Check if files exist
    if not os.path.exists(preprocessor.train_file):
        print(f"\nERROR - ERROR: Training file not found: {preprocessor.train_file}")
        sys.exit(1)

    if not os.path.exists(preprocessor.test_file):
        print(f"\nERROR - ERROR: Test file not found: {preprocessor.test_file}")
        sys.exit(1)

    if not os.path.exists(preprocessor.test_labels_file):
        print(f"\nERROR - ERROR: Test labels file not found: {preprocessor.test_labels_file}")
        sys.exit(1)

    # Run preprocessing
    try:
        X_train, X_test, y_train, y_test = preprocessor.preprocess_hai_dataset()

        print("\n" + "="*80)
        print("OK - SUCCESS: HAI dataset ready for experiments!")
        print("="*80)
        print(f" End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

    except Exception as e:
        print(f"\nERROR - ERROR: Preprocessing failed")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
