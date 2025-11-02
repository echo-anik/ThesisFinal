"""
FINAL HAI HIERARCHICAL IDS EXPERIMENTS
======================================
Comprehensive testing with all edge percentages: 5%, 10%, 15%, 20%, 25%
Central model uses 100% training data (trained once per config)
Detailed metrics: TP, FP, FN, TN, precision, recall, F1, accuracy
Power efficiency and Raspberry Pi resource analysis
"""

import pandas as pd
import numpy as np
import os
import pickle
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')


class FinalHAIExperimentRunner:
    def __init__(self, base_dir='g:\\THESIS'):
        self.base_dir = base_dir
        self.processed_dir = os.path.join(base_dir, 'processed_data')
        self.results_dir = os.path.join(base_dir, 'Results', 'HAI')
        os.makedirs(self.results_dir, exist_ok=True)

    def load_hai_data(self):
        """Load HAI dataset"""
        X_train = pd.read_csv(os.path.join(self.processed_dir, 'hai_train_fixed.csv')).values
        X_test = pd.read_csv(os.path.join(self.processed_dir, 'hai_test_fixed.csv')).values
        
        y_train_file = pd.read_csv(os.path.join(self.processed_dir, 'hai_train_labels_fixed.csv'))
        y_test_file = pd.read_csv(os.path.join(self.processed_dir, 'hai_test_labels_fixed.csv'))
        
        if 'attack' in y_train_file.columns:
            y_train = y_train_file['attack'].values
            y_test = y_test_file['attack'].values
        else:
            y_train = y_train_file.iloc[:, 0].values
            y_test = y_test_file.iloc[:, 0].values
        
        y_train = np.asarray(y_train, dtype=int)
        y_test = np.asarray(y_test, dtype=int)
        
        return X_train, X_test, y_train, y_test

    def get_6_configurations(self):
        """Return 6 top model configurations"""
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
        return configs

    def measure_model_size(self, model):
        """Measure model size in MB"""
        return len(pickle.dumps(model)) / (1024 * 1024)

    def measure_inference_time(self, model, X_test, n_runs=3):
        """Measure average inference time in milliseconds"""
        times = []
        for _ in range(n_runs):
            start = time.time()
            _ = model.predict(X_test)
            times.append((time.time() - start) * 1000)
        return np.mean(times)

    def estimate_pi_power(self, model_size_mb, inference_time_ms, num_features):
        """Estimate power for Raspberry Pi 4"""
        base_power_w = 2.0
        memory_factor = 0.001
        cpu_factor = 0.001
        return base_power_w + (model_size_mb * memory_factor) + (inference_time_ms * cpu_factor)

    def run_experiment(self, X_train, X_test, y_train, y_test, 
                      edge_model_template, central_model, config_name, edge_pct):
        """Run single hierarchical experiment"""
        from sklearn.base import clone
        
        # Sample for edge
        sample_size = int(len(X_train) * edge_pct)
        X_edge, y_edge = resample(X_train, y_train, n_samples=sample_size, 
                                 random_state=42, stratify=y_train)
        
        # Train edge
        edge_model = clone(edge_model_template)
        edge_model.fit(X_edge, y_edge)
        
        # Edge predictions
        edge_pred = edge_model.predict(X_test)
        edge_proba = edge_model.predict_proba(X_test)[:, 1]
        
        # Escalation logic: attack prediction OR uncertain
        edge_uncertain = (edge_proba > 0.4) & (edge_proba < 0.6)
        escalate_mask = (edge_pred == 1) | edge_uncertain
        escalation_count = escalate_mask.sum()
        escalation_pct = escalation_count / len(X_test) * 100
        
        # Final predictions
        final_pred = edge_pred.copy()
        if escalation_count > 0:
            central_pred = central_model.predict(X_test[escalate_mask])
            final_pred[escalate_mask] = central_pred
        
        # Metrics
        tn, fp, fn, tp = confusion_matrix(y_test, final_pred).ravel()
        
        final_f1 = f1_score(y_test, final_pred, zero_division=0)
        final_precision = precision_score(y_test, final_pred, zero_division=0)
        final_recall = recall_score(y_test, final_pred, zero_division=0)
        final_accuracy = accuracy_score(y_test, final_pred)
        
        # Resources
        edge_size = self.measure_model_size(edge_model)
        edge_time = self.measure_inference_time(edge_model, X_test)
        edge_power = self.estimate_pi_power(edge_size, edge_time, X_test.shape[1])
        
        # Edge-only performance
        edge_f1 = f1_score(y_test, edge_pred, zero_division=0)
        edge_cm = confusion_matrix(y_test, edge_pred)
        if edge_cm.size == 4:
            edge_tn, edge_fp, edge_fn, edge_tp = edge_cm.ravel()
        else:
            edge_tn = edge_cm[0, 0] if edge_cm.shape[0] > 0 and edge_cm.shape[1] > 0 else 0
            edge_tp = edge_cm[1, 1] if edge_cm.shape[0] > 1 and edge_cm.shape[1] > 1 else 0
            edge_fp = edge_cm[0, 1] if edge_cm.shape[0] > 0 and edge_cm.shape[1] > 1 else 0
            edge_fn = edge_cm[1, 0] if edge_cm.shape[0] > 1 and edge_cm.shape[1] > 0 else 0
        
        return {
            'config_name': config_name,
            'edge_pct': edge_pct,
            'edge_samples': sample_size,
            'edge_f1': edge_f1,
            'edge_tp': int(edge_tp),
            'edge_fp': int(edge_fp),
            'edge_fn': int(edge_fn),
            'edge_tn': int(edge_tn),
            'final_f1': final_f1,
            'final_precision': final_precision,
            'final_recall': final_recall,
            'final_accuracy': final_accuracy,
            'final_tp': int(tp),
            'final_fp': int(fp),
            'final_fn': int(fn),
            'final_tn': int(tn),
            'escalation_count': escalation_count,
            'escalation_pct': escalation_pct,
            'model_size_mb': edge_size,
            'inference_ms': edge_time,
            'pi_power_w': edge_power,
            'timestamp': datetime.now().isoformat()
        }

    def run_all_experiments(self):
        """Run comprehensive experiments"""
        print("\n" + "="*80)
        print("FINAL HAI HIERARCHICAL IDS EXPERIMENTS")
        print("="*80)
        print("6 Configurations x 5 Edge Percentages x Detailed Metrics")
        print("="*80)
        
        X_train, X_test, y_train, y_test = self.load_hai_data()
        
        print(f"\nDataset loaded:")
        print(f"  Train: {X_train.shape} ({(y_train==1).sum():,} attacks, {(y_train==1).sum()/len(y_train)*100:.1f}%)")
        print(f"  Test: {X_test.shape} ({(y_test==1).sum():,} attacks, {(y_test==1).sum()/len(y_test)*100:.2f}%)")
        
        all_results = []
        configs = self.get_6_configurations()
        edge_percentages = [0.05, 0.10, 0.15, 0.20, 0.25]
        
        for config in configs:
            print(f"\n{'#'*80}")
            print(f"Configuration: {config['name']}")
            print(f"{'#'*80}")
            
            # Train central once (using 100% data)
            print("Training central model (100% training data)...")
            central_model = config['central']
            central_model.fit(X_train, y_train)
            central_size = self.measure_model_size(central_model)
            print(f"  Central model size: {central_size:.2f} MB")
            
            for edge_pct in edge_percentages:
                print(f"\n  Edge {int(edge_pct*100):2d}%:", end=" ")
                
                try:
                    result = self.run_experiment(
                        X_train, X_test, y_train, y_test,
                        config['edge'], central_model,
                        config['name'], edge_pct
                    )
                    all_results.append(result)
                    
                    print(f"F1={result['final_f1']:.4f} TP={result['final_tp']:5d} FP={result['final_fp']:4d} " +
                          f"FN={result['final_fn']:3d} Power={result['pi_power_w']:.2f}W Esc={result['escalation_pct']:5.1f}%")
                
                except Exception as e:
                    print(f"ERROR - {e}")
        
        if all_results:
            self.save_and_summarize(all_results)

    def save_and_summarize(self, results):
        """Save results and print summary"""
        df = pd.DataFrame(results)
        
        # Save detailed results
        results_file = os.path.join(self.results_dir, 'HAI_FINAL_DETAILED_RESULTS.csv')
        df.to_csv(results_file, index=False)
        print(f"\n\n{'='*80}")
        print(f"Results saved to: {results_file}")
        print(f"Total experiments: {len(results)}")
        print(f"{'='*80}")
        
        # Print summary by configuration
        print("\n" + "="*80)
        print("SUMMARY BY CONFIGURATION")
        print("="*80)
        
        for config_name in sorted(df['config_name'].unique()):
            config_data = df[df['config_name'] == config_name]
            best_idx = config_data['final_f1'].idxmax()
            best = df.iloc[best_idx]
            
            print(f"\n{config_name}:")
            print(f"  Best F1: {best['final_f1']:.4f} (at {int(best['edge_pct']*100)}% edge)")
            print(f"  Metrics: Precision={best['final_precision']:.4f}, Recall={best['final_recall']:.4f}, Accuracy={best['final_accuracy']:.4f}")
            print(f"  Confusion Matrix: TP={best['final_tp']:5d} FP={best['final_fp']:4d} FN={best['final_fn']:3d} TN={best['final_tn']:6d}")
            print(f"  Escalation: {best['escalation_pct']:.1f}% ({int(best['escalation_count'])} samples)")
            print(f"  Resources: Model={best['model_size_mb']:.2f}MB, Inference={best['inference_ms']:.1f}ms, Power={best['pi_power_w']:.3f}W")
            
            # Show all edge percentages for this config
            print(f"  Edge percentages:")
            for _, row in config_data.iterrows():
                print(f"    {int(row['edge_pct']*100):2d}%: F1={row['final_f1']:.4f}, Esc={row['escalation_pct']:5.1f}%, TP={row['final_tp']:5d}")
        
        # Overall best
        print(f"\n\n{'='*80}")
        print("BEST OVERALL")
        print(f"{'='*80}")
        best_idx = df['final_f1'].idxmax()
        best = df.iloc[best_idx]
        print(f"\nConfiguration: {best['config_name']}")
        print(f"Edge Percentage: {int(best['edge_pct']*100)}%")
        print(f"\nACCURACY METRICS:")
        print(f"  F1 Score: {best['final_f1']:.4f}")
        print(f"  Precision: {best['final_precision']:.4f}")
        print(f"  Recall: {best['final_recall']:.4f}")
        print(f"  Accuracy: {best['final_accuracy']:.4f}")
        print(f"\nCONFUSION MATRIX:")
        print(f"  True Positives (TP): {best['final_tp']:,}")
        print(f"  False Positives (FP): {best['final_fp']:,}")
        print(f"  False Negatives (FN): {best['final_fn']:,}")
        print(f"  True Negatives (TN): {best['final_tn']:,}")
        print(f"\nEDGE PERFORMANCE:")
        print(f"  Edge Model Alone F1: {best['edge_f1']:.4f}")
        print(f"  Improvement with Central: +{(best['final_f1']-best['edge_f1'])*100:.2f}% absolute")
        print(f"\nRESPURCE EFFICIENCY:")
        print(f"  Model Size: {best['model_size_mb']:.2f} MB")
        print(f"  Inference Time: {best['inference_ms']:.1f} ms")
        print(f"  Raspberry Pi Power: {best['pi_power_w']:.3f} W")
        print(f"  Escalation Rate: {best['escalation_pct']:.1f}% ({int(best['escalation_count'])} of {len(results)} samples)")


if __name__ == '__main__':
    runner = FinalHAIExperimentRunner()
    runner.run_all_experiments()
