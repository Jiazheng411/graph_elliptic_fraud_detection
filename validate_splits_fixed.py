"""
Fixed validation script for the created data splits.
"""

import pandas as pd
import numpy as np
import os

def validate_splits():
    """Validate that the splits are correct and show basic statistics."""
    print("Bitcoin Fraud Detection - Split Validation (Fixed)")
    print("=" * 55)
    
    # Check labeled-only splits
    print("\n1. LABELED-ONLY SPLITS (for traditional ML)")
    print("-" * 45)
    
    total_labeled_txs = 0
    total_illicit = 0
    total_licit = 0
    
    for split in ['train', 'val', 'test']:
        features_file = f'splits/labeled_only/{split}_features.csv'
        classes_file = f'splits/labeled_only/{split}_classes.csv'
        
        if os.path.exists(features_file) and os.path.exists(classes_file):
            features_df = pd.read_csv(features_file)
            classes_df = pd.read_csv(classes_file)
            
            # Check timestep ranges
            timestep_range = (features_df['timestep'].min(), features_df['timestep'].max())
            
            # Check class distribution - ensure string comparison
            classes_df['class'] = classes_df['class'].astype(str)
            illicit_count = len(classes_df[classes_df['class'] == '1'])
            licit_count = len(classes_df[classes_df['class'] == '2'])
            
            total_labeled_txs += len(features_df)
            total_illicit += illicit_count
            total_licit += licit_count
            
            print(f"{split.upper():5}: {len(features_df):6,} txs | Timesteps {timestep_range[0]:2d}-{timestep_range[1]:2d} | Illicit: {illicit_count:4,} | Licit: {licit_count:5,}")
    
    print(f"{'TOTAL':5}: {total_labeled_txs:6,} txs | Illicit: {total_illicit:4,} | Licit: {total_licit:5,}")
    
    # Check full dataset splits
    print("\n2. FULL DATASET SPLITS (for graph methods)")
    print("-" * 45)
    
    total_full_txs = 0
    total_full_illicit = 0
    total_full_licit = 0
    total_unknown = 0
    
    for split in ['train', 'val', 'test']:
        features_file = f'splits/full_dataset/{split}_features.csv'
        classes_file = f'splits/full_dataset/{split}_classes.csv'
        
        if os.path.exists(features_file) and os.path.exists(classes_file):
            features_df = pd.read_csv(features_file)
            classes_df = pd.read_csv(classes_file)
            
            # Check timestep ranges
            timestep_range = (features_df['timestep'].min(), features_df['timestep'].max())
            
            # Check class distribution
            classes_df['class'] = classes_df['class'].astype(str)
            illicit_count = len(classes_df[classes_df['class'] == '1'])
            licit_count = len(classes_df[classes_df['class'] == '2'])
            unknown_count = len(classes_df[classes_df['class'] == 'unknown'])
            
            total_full_txs += len(features_df)
            total_full_illicit += illicit_count
            total_full_licit += licit_count
            total_unknown += unknown_count
            
            print(f"{split.upper():5}: {len(features_df):6,} txs | Timesteps {timestep_range[0]:2d}-{timestep_range[1]:2d} | Illicit: {illicit_count:4,} | Licit: {licit_count:5,} | Unknown: {unknown_count:6,}")
    
    print(f"{'TOTAL':5}: {total_full_txs:6,} txs | Illicit: {total_full_illicit:4,} | Licit: {total_full_licit:5,} | Unknown: {total_unknown:6,}")

def show_class_balance():
    """Show class balance statistics."""
    print("\n3. CLASS BALANCE ANALYSIS")
    print("-" * 45)
    
    # Labeled splits
    for split in ['train', 'val', 'test']:
        classes_file = f'splits/labeled_only/{split}_classes.csv'
        if os.path.exists(classes_file):
            classes_df = pd.read_csv(classes_file)
            classes_df['class'] = classes_df['class'].astype(str)
            
            illicit = len(classes_df[classes_df['class'] == '1'])
            total = len(classes_df)
            if total > 0:
                illicit_pct = illicit / total * 100
                print(f"Labeled {split:5}: {illicit_pct:5.1f}% illicit ({illicit:,} / {total:,})")
    
    # Full dataset splits (only labeled portion)
    print("\nFull dataset (labeled portion only):")
    for split in ['train', 'val', 'test']:
        classes_file = f'splits/full_dataset/{split}_classes.csv'
        if os.path.exists(classes_file):
            classes_df = pd.read_csv(classes_file)
            classes_df['class'] = classes_df['class'].astype(str)
            
            # Only count labeled transactions
            labeled_df = classes_df[classes_df['class'].isin(['1', '2'])]
            illicit = len(labeled_df[labeled_df['class'] == '1'])
            total_labeled = len(labeled_df)
            total_all = len(classes_df)
            
            if total_labeled > 0:
                illicit_pct = illicit / total_labeled * 100
                labeled_pct = total_labeled / total_all * 100
                print(f"Full {split:5}: {illicit_pct:5.1f}% illicit | {labeled_pct:5.1f}% labeled ({total_labeled:,} / {total_all:,})")

def demo_usage():
    """Show usage examples."""
    print("\n4. USAGE EXAMPLES")
    print("-" * 45)
    
    print("\nFor XGBoost (labeled-only):")
    print("```python")
    print("import pandas as pd")
    print("import xgboost as xgb")
    print("")
    print("# Load training data")
    print("train_features = pd.read_csv('splits/labeled_only/train_features.csv')")
    print("train_classes = pd.read_csv('splits/labeled_only/train_classes.csv')")
    print("")
    print("# Prepare features and labels")
    print("X_train = train_features.iloc[:, 2:].values  # Skip txId and timestep") 
    print("y_train = train_classes['class'].map({'1': 1, '2': 0}).values")
    print("")
    print("# Train XGBoost")
    print("clf = xgb.XGBClassifier(n_estimators=100, max_depth=6)")
    print("clf.fit(X_train, y_train)")
    print("```")
    
    print("\nFor Graph Neural Networks (full dataset):")
    print("```python")
    print("import pandas as pd")
    print("import torch")
    print("from torch_geometric.data import Data")
    print("")
    print("# Load training data")
    print("train_features = pd.read_csv('splits/full_dataset/train_features.csv')")
    print("train_classes = pd.read_csv('splits/full_dataset/train_classes.csv')")
    print("train_edges = pd.read_csv('splits/full_dataset/train_edges.csv')")
    print("")
    print("# Prepare node features")
    print("features = train_features.set_index('txId').iloc[:, 1:].values")
    print("")
    print("# Prepare labels (-1 for unknown)")
    print("labels = train_classes.set_index('txId')['class'].map({")
    print("    '1': 1, '2': 0, 'unknown': -1")
    print("})")
    print("```")

if __name__ == "__main__":
    validate_splits()
    show_class_balance()
    demo_usage()
    
    print("\n" + "=" * 55)
    print("Validation complete! Your splits are ready for experiments.")
    print("Share the 'splits/' folder with your team members.")
    print("See DATA_SPLITTING_README.md for detailed documentation.")