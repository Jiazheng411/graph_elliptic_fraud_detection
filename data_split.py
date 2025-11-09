"""
Bitcoin Transaction Data Splitter
This script creates train/test splits for the Elliptic Bitcoin dataset based on temporal information.

Two types of splits are created:
1. Labeled-only: Contains only transactions with known labels (1=illicit, 2=licit)
2. Full dataset: Contains all transactions including unknown labels

Temporal split:
- Train: Time steps 1-29
- Validation: Time steps 31-34 
- Test: Time steps 35-49
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any
import json
from datetime import datetime

class EllipticDataSplitter:
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data splitter.
        
        Args:
            data_dir: Directory containing the CSV files
        """
        self.data_dir = data_dir
        self.features_file = os.path.join(data_dir, "elliptic_txs_features.csv")
        self.classes_file = os.path.join(data_dir, "elliptic_txs_classes.csv")
        self.edgelist_file = os.path.join(data_dir, "elliptic_txs_edgelist.csv")
        
        # Split definitions
        self.train_timesteps = list(range(1, 30))  # 1-29
        self.val_timesteps = list(range(31, 35))   # 31-34
        self.test_timesteps = list(range(35, 50))  # 35-49
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all data files.
        
        Returns:
            Tuple of (features_df, classes_df, edges_df)
        """
        print("Loading data files...")
        
        # Load features (large file, so we'll be careful about memory)
        print("Loading features...")
        features_df = pd.read_csv(self.features_file, header=None)
        
        # Set column names - first column is txId, second is timestep, rest are features
        feature_cols = ['txId', 'timestep'] + [f'feature_{i}' for i in range(1, features_df.shape[1] - 1)]
        features_df.columns = feature_cols
        
        print("Loading classes...")
        classes_df = pd.read_csv(self.classes_file)
        
        print("Loading edge list...")
        edges_df = pd.read_csv(self.edgelist_file)
        
        print(f"Loaded {len(features_df)} transactions, {len(classes_df)} class labels, {len(edges_df)} edges")
        
        return features_df, classes_df, edges_df
    
    def get_timestep_stats(self, features_df: pd.DataFrame, classes_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about the temporal distribution of data.
        
        Args:
            features_df: Features dataframe
            classes_df: Classes dataframe
            
        Returns:
            Dictionary with statistics
        """
        # Merge to get timestep info for each transaction
        merged_df = features_df[['txId', 'timestep']].merge(classes_df, on='txId', how='left')
        
        stats = {
            'timestep_range': (features_df['timestep'].min(), features_df['timestep'].max()),
            'total_transactions': len(features_df),
            'labeled_transactions': len(classes_df[classes_df['class'] != 'unknown']),
            'illicit_transactions': len(classes_df[classes_df['class'] == '1']),
            'licit_transactions': len(classes_df[classes_df['class'] == '2']),
            'unknown_transactions': len(classes_df[classes_df['class'] == 'unknown']),
            'timestep_distribution': {}
        }
        
        # Get distribution by timestep
        for timestep in sorted(features_df['timestep'].unique()):
            timestep_data = merged_df[merged_df['timestep'] == timestep]
            stats['timestep_distribution'][timestep] = {
                'total': len(timestep_data),
                'illicit': len(timestep_data[timestep_data['class'] == '1']),
                'licit': len(timestep_data[timestep_data['class'] == '2']),
                'unknown': len(timestep_data[timestep_data['class'] == 'unknown'])
            }
        
        return stats
    
    def create_labeled_split(self, features_df: pd.DataFrame, classes_df: pd.DataFrame, 
                           edges_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create train/val/test split with only labeled data (classes 1 and 2).
        
        Args:
            features_df: Features dataframe
            classes_df: Classes dataframe  
            edges_df: Edges dataframe
            
        Returns:
            Dictionary with split dataframes
        """
        print("Creating labeled-only split...")
        
        # Filter to only labeled transactions
        labeled_classes = classes_df[classes_df['class'].isin(['1', '2'])].copy()
        
        # Merge with features to get timestep info
        labeled_features = features_df.merge(labeled_classes[['txId']], on='txId', how='inner')
        
        # Split by timestep
        train_features = labeled_features[labeled_features['timestep'].isin(self.train_timesteps)]
        val_features = labeled_features[labeled_features['timestep'].isin(self.val_timesteps)]
        test_features = labeled_features[labeled_features['timestep'].isin(self.test_timesteps)]
        
        # Get corresponding class labels
        train_classes = labeled_classes[labeled_classes['txId'].isin(train_features['txId'])]
        val_classes = labeled_classes[labeled_classes['txId'].isin(val_features['txId'])]
        test_classes = labeled_classes[labeled_classes['txId'].isin(test_features['txId'])]
        
        # Filter edges to only include transactions in our splits
        all_labeled_txids = set(labeled_features['txId'])
        labeled_edges = edges_df[
            (edges_df['txId1'].isin(all_labeled_txids)) & 
            (edges_df['txId2'].isin(all_labeled_txids))
        ]
        
        # Create separate edge lists for each split
        train_txids = set(train_features['txId'])
        val_txids = set(val_features['txId'])
        test_txids = set(test_features['txId'])
        
        train_edges = labeled_edges[
            (labeled_edges['txId1'].isin(train_txids)) & 
            (labeled_edges['txId2'].isin(train_txids))
        ]
        
        val_edges = labeled_edges[
            (labeled_edges['txId1'].isin(val_txids)) & 
            (labeled_edges['txId2'].isin(val_txids))
        ]
        
        test_edges = labeled_edges[
            (labeled_edges['txId1'].isin(test_txids)) & 
            (labeled_edges['txId2'].isin(test_txids))
        ]
        
        print(f"Labeled split - Train: {len(train_features)} txs, Val: {len(val_features)} txs, Test: {len(test_features)} txs")
        print(f"Labeled edges - Train: {len(train_edges)}, Val: {len(val_edges)}, Test: {len(test_edges)}")
        
        return {
            'train_features': train_features,
            'train_classes': train_classes,
            'train_edges': train_edges,
            'val_features': val_features,
            'val_classes': val_classes,
            'val_edges': val_edges,
            'test_features': test_features,
            'test_classes': test_classes,
            'test_edges': test_edges,
            'all_labeled_edges': labeled_edges
        }
    
    def create_full_split(self, features_df: pd.DataFrame, classes_df: pd.DataFrame, 
                         edges_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create train/val/test split with all data including unknown labels.
        
        Args:
            features_df: Features dataframe
            classes_df: Classes dataframe
            edges_df: Edges dataframe
            
        Returns:
            Dictionary with split dataframes
        """
        print("Creating full dataset split...")
        
        # Split features by timestep
        train_features = features_df[features_df['timestep'].isin(self.train_timesteps)]
        val_features = features_df[features_df['timestep'].isin(self.val_timesteps)]
        test_features = features_df[features_df['timestep'].isin(self.test_timesteps)]
        
        # Get corresponding class labels (including unknown)
        train_classes = classes_df[classes_df['txId'].isin(train_features['txId'])]
        val_classes = classes_df[classes_df['txId'].isin(val_features['txId'])]
        test_classes = classes_df[classes_df['txId'].isin(test_features['txId'])]
        
        # Create separate edge lists for each split
        train_txids = set(train_features['txId'])
        val_txids = set(val_features['txId'])
        test_txids = set(test_features['txId'])
        
        train_edges = edges_df[
            (edges_df['txId1'].isin(train_txids)) & 
            (edges_df['txId2'].isin(train_txids))
        ]
        
        val_edges = edges_df[
            (edges_df['txId1'].isin(val_txids)) & 
            (edges_df['txId2'].isin(val_txids))
        ]
        
        test_edges = edges_df[
            (edges_df['txId1'].isin(test_txids)) & 
            (edges_df['txId2'].isin(test_txids))
        ]
        
        print(f"Full split - Train: {len(train_features)} txs, Val: {len(val_features)} txs, Test: {len(test_features)} txs")
        print(f"Full edges - Train: {len(train_edges)}, Val: {len(val_edges)}, Test: {len(test_edges)}")
        
        return {
            'train_features': train_features,
            'train_classes': train_classes,
            'train_edges': train_edges,
            'val_features': val_features,
            'val_classes': val_classes,
            'val_edges': val_edges,
            'test_features': test_features,
            'test_classes': test_classes,
            'test_edges': test_edges,
            'all_edges': edges_df
        }
    
    def save_splits(self, labeled_split: Dict[str, pd.DataFrame], 
                   full_split: Dict[str, pd.DataFrame], output_dir: str = "splits"):
        """
        Save the split data to CSV files.
        
        Args:
            labeled_split: Dictionary with labeled split dataframes
            full_split: Dictionary with full split dataframes
            output_dir: Directory to save splits
        """
        # Create output directories
        labeled_dir = os.path.join(output_dir, "labeled_only")
        full_dir = os.path.join(output_dir, "full_dataset")
        
        os.makedirs(labeled_dir, exist_ok=True)
        os.makedirs(full_dir, exist_ok=True)
        
        print(f"Saving splits to {output_dir}...")
        
        # Save labeled splits
        for split_name in ['train', 'val', 'test']:
            labeled_split[f'{split_name}_features'].to_csv(
                os.path.join(labeled_dir, f'{split_name}_features.csv'), index=False
            )
            labeled_split[f'{split_name}_classes'].to_csv(
                os.path.join(labeled_dir, f'{split_name}_classes.csv'), index=False
            )
            labeled_split[f'{split_name}_edges'].to_csv(
                os.path.join(labeled_dir, f'{split_name}_edges.csv'), index=False
            )
        
        # Save all labeled edges
        labeled_split['all_labeled_edges'].to_csv(
            os.path.join(labeled_dir, 'all_edges.csv'), index=False
        )
        
        # Save full splits
        for split_name in ['train', 'val', 'test']:
            full_split[f'{split_name}_features'].to_csv(
                os.path.join(full_dir, f'{split_name}_features.csv'), index=False
            )
            full_split[f'{split_name}_classes'].to_csv(
                os.path.join(full_dir, f'{split_name}_classes.csv'), index=False
            )
            full_split[f'{split_name}_edges'].to_csv(
                os.path.join(full_dir, f'{split_name}_edges.csv'), index=False
            )
        
        # Save all edges for full dataset
        full_split['all_edges'].to_csv(
            os.path.join(full_dir, 'all_edges.csv'), index=False
        )
        
        print("Splits saved successfully!")
    
    def save_metadata(self, stats: Dict[str, Any], labeled_split: Dict[str, pd.DataFrame],
                     full_split: Dict[str, pd.DataFrame], output_dir: str = "splits"):
        """
        Save metadata about the splits.
        
        Args:
            stats: Statistics about the dataset
            labeled_split: Labeled split data
            full_split: Full split data
            output_dir: Directory to save metadata
        """
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'split_criteria': {
                'train_timesteps': self.train_timesteps,
                'val_timesteps': self.val_timesteps,
                'test_timesteps': self.test_timesteps
            },
            'dataset_stats': stats,
            'labeled_split_stats': {
                'train': {
                    'num_transactions': len(labeled_split['train_features']),
                    'num_edges': len(labeled_split['train_edges']),
                    'illicit_count': len(labeled_split['train_classes'][labeled_split['train_classes']['class'] == '1']),
                    'licit_count': len(labeled_split['train_classes'][labeled_split['train_classes']['class'] == '2'])
                },
                'val': {
                    'num_transactions': len(labeled_split['val_features']),
                    'num_edges': len(labeled_split['val_edges']),
                    'illicit_count': len(labeled_split['val_classes'][labeled_split['val_classes']['class'] == '1']),
                    'licit_count': len(labeled_split['val_classes'][labeled_split['val_classes']['class'] == '2'])
                },
                'test': {
                    'num_transactions': len(labeled_split['test_features']),
                    'num_edges': len(labeled_split['test_edges']),
                    'illicit_count': len(labeled_split['test_classes'][labeled_split['test_classes']['class'] == '1']),
                    'licit_count': len(labeled_split['test_classes'][labeled_split['test_classes']['class'] == '2'])
                }
            },
            'full_split_stats': {
                'train': {
                    'num_transactions': len(full_split['train_features']),
                    'num_edges': len(full_split['train_edges']),
                    'illicit_count': len(full_split['train_classes'][full_split['train_classes']['class'] == '1']),
                    'licit_count': len(full_split['train_classes'][full_split['train_classes']['class'] == '2']),
                    'unknown_count': len(full_split['train_classes'][full_split['train_classes']['class'] == 'unknown'])
                },
                'val': {
                    'num_transactions': len(full_split['val_features']),
                    'num_edges': len(full_split['val_edges']),
                    'illicit_count': len(full_split['val_classes'][full_split['val_classes']['class'] == '1']),
                    'licit_count': len(full_split['val_classes'][full_split['val_classes']['class'] == '2']),
                    'unknown_count': len(full_split['val_classes'][full_split['val_classes']['class'] == 'unknown'])
                },
                'test': {
                    'num_transactions': len(full_split['test_features']),
                    'num_edges': len(full_split['test_edges']),
                    'illicit_count': len(full_split['test_classes'][full_split['test_classes']['class'] == '1']),
                    'licit_count': len(full_split['test_classes'][full_split['test_classes']['class'] == '2']),
                    'unknown_count': len(full_split['test_classes'][full_split['test_classes']['class'] == 'unknown'])
                }
            }
        }
        
        # Save metadata (convert numpy types to native Python types for JSON serialization)
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif hasattr(obj, 'item'):  # other numpy scalars
                return obj.item()
            else:
                return obj
        
        metadata_serializable = convert_numpy_types(metadata)
        
        with open(os.path.join(output_dir, 'split_metadata.json'), 'w') as f:
            json.dump(metadata_serializable, f, indent=2)
        
        print("Metadata saved to split_metadata.json")
    
    def run_split(self, output_dir: str = "splits"):
        """
        Run the complete data splitting process.
        
        Args:
            output_dir: Directory to save the splits
        """
        print("Starting Bitcoin transaction data splitting...")
        print("=" * 50)
        
        # Load data
        features_df, classes_df, edges_df = self.load_data()
        
        # Get statistics
        stats = self.get_timestep_stats(features_df, classes_df)
        
        print(f"\nDataset Overview:")
        print(f"Timestep range: {stats['timestep_range']}")
        print(f"Total transactions: {stats['total_transactions']:,}")
        print(f"Labeled transactions: {stats['labeled_transactions']:,}")
        print(f"  - Illicit: {stats['illicit_transactions']:,}")
        print(f"  - Licit: {stats['licit_transactions']:,}")
        print(f"Unknown transactions: {stats['unknown_transactions']:,}")
        
        # Create splits
        print(f"\nCreating temporal splits:")
        print(f"Train timesteps: {self.train_timesteps}")
        print(f"Validation timesteps: {self.val_timesteps}")
        print(f"Test timesteps: {self.test_timesteps}")
        
        labeled_split = self.create_labeled_split(features_df, classes_df, edges_df)
        full_split = self.create_full_split(features_df, classes_df, edges_df)
        
        # Save everything
        self.save_splits(labeled_split, full_split, output_dir)
        self.save_metadata(stats, labeled_split, full_split, output_dir)
        
        print(f"\n✅ Data splitting complete!")
        print(f"Check the '{output_dir}' directory for your split files.")
        print(f"  - '{output_dir}/labeled_only/': Contains only transactions with known labels")
        print(f"  - '{output_dir}/full_dataset/': Contains all transactions including unknown labels")
        print(f"  - '{output_dir}/split_metadata.json': Contains detailed statistics and metadata")


def main():
    """Main function to run the data splitting."""
    splitter = EllipticDataSplitter(data_dir="data")
    splitter.run_split(output_dir="splits")


if __name__ == "__main__":
    main()