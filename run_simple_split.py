"""
Simple data splitting script - no metadata to avoid JSON issues
"""

from data_split import EllipticDataSplitter
import pandas as pd
import os

def run_simple_split():
    """Run the data splitting without metadata."""
    print("Bitcoin Fraud Detection - Data Splitting (Simple)")
    print("=" * 50)
    
    # Initialize splitter
    splitter = EllipticDataSplitter(data_dir="data")
    
    # Load data
    features_df, classes_df, edges_df = splitter.load_data()
    
    # Get basic stats
    stats = splitter.get_timestep_stats(features_df, classes_df)
    
    print(f"\nDataset Overview:")
    print(f"Timestep range: {stats['timestep_range']}")
    print(f"Total transactions: {stats['total_transactions']:,}")
    print(f"Labeled transactions: {stats['labeled_transactions']:,}")
    print(f"  - Illicit: {stats['illicit_transactions']:,}")
    print(f"  - Licit: {stats['licit_transactions']:,}")
    print(f"Unknown transactions: {stats['unknown_transactions']:,}")
    
    # Create splits
    print(f"\nCreating temporal splits:")
    print(f"Train timesteps: {splitter.train_timesteps}")
    print(f"Validation timesteps: {splitter.val_timesteps}")
    print(f"Test timesteps: {splitter.test_timesteps}")
    
    labeled_split = splitter.create_labeled_split(features_df, classes_df, edges_df)
    full_split = splitter.create_full_split(features_df, classes_df, edges_df)
    
    # Save splits only
    splitter.save_splits(labeled_split, full_split, "splits")
    
    print(f"\nData splitting complete!")
    print(f"Files saved in 'splits/' directory:")
    print(f"  - 'splits/labeled_only/': Contains only transactions with known labels")
    print(f"  - 'splits/full_dataset/': Contains all transactions including unknown labels")

def show_split_summary():
    """Show summary of created splits."""
    print("\n" + "=" * 50)
    print("SPLIT SUMMARY")
    print("=" * 50)
    
    # Check labeled splits
    if os.path.exists('splits/labeled_only'):
        print("\nLabeled-only splits:")
        for split in ['train', 'val', 'test']:
            features_file = f'splits/labeled_only/{split}_features.csv'
            classes_file = f'splits/labeled_only/{split}_classes.csv'
            edges_file = f'splits/labeled_only/{split}_edges.csv'
            
            if all(os.path.exists(f) for f in [features_file, classes_file, edges_file]):
                features_df = pd.read_csv(features_file)
                classes_df = pd.read_csv(classes_file)
                edges_df = pd.read_csv(edges_file)
                
                illicit_count = len(classes_df[classes_df['class'] == '1'])
                licit_count = len(classes_df[classes_df['class'] == '2'])
                
                print(f"  {split.capitalize()}: {len(features_df):,} txs, {len(edges_df):,} edges")
                print(f"    - Illicit: {illicit_count:,}, Licit: {licit_count:,}")
    
    # Check full splits
    if os.path.exists('splits/full_dataset'):
        print("\nFull dataset splits:")
        for split in ['train', 'val', 'test']:
            features_file = f'splits/full_dataset/{split}_features.csv'
            classes_file = f'splits/full_dataset/{split}_classes.csv'
            edges_file = f'splits/full_dataset/{split}_edges.csv'
            
            if all(os.path.exists(f) for f in [features_file, classes_file, edges_file]):
                features_df = pd.read_csv(features_file)
                classes_df = pd.read_csv(classes_file)
                edges_df = pd.read_csv(edges_file)
                
                illicit_count = len(classes_df[classes_df['class'] == '1'])
                licit_count = len(classes_df[classes_df['class'] == '2'])
                unknown_count = len(classes_df[classes_df['class'] == 'unknown'])
                
                print(f"  {split.capitalize()}: {len(features_df):,} txs, {len(edges_df):,} edges")
                print(f"    - Illicit: {illicit_count:,}, Licit: {licit_count:,}, Unknown: {unknown_count:,}")

if __name__ == "__main__":
    run_simple_split()
    show_split_summary()
    
    print("\nReady for your experiments!")
    print("Use the DATA_SPLITTING_README.md for usage examples.")