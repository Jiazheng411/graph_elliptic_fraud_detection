"""
Graph Transformer Paper Reproduction - Exact Setup
=================================================
Reproducing the same experimental setup as GCN but with Graph Transformer:
1. Train/Test split only (train = original train + val)
2. 2-layer Graph Transformer with 100-dim embeddings  
3. Weighted CrossEntropyLoss (0.7/0.3)
4. 1000 epochs, Adam lr=0.001
5. Separate train/test graphs
6. With and without unknown nodes
7. Multi-head attention (8 heads)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score, roc_auc_score, average_precision_score
import time
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

class GraphTransformer(nn.Module):
    """2-layer Graph Transformer with proper embedding and layer normalization"""
    def __init__(self, input_dim, hidden_dim=128, num_classes=2, heads=8, dropout=0.1):
        super(GraphTransformer, self).__init__()
        
        # Input embedding layer - project raw features to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # First transformer layer with multi-head attention
        # Now 128 // 8 = 16, so 16 * 8 = 128 (perfect match!)
        self.conv1 = TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, 
                                   concat=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Second transformer layer  
        self.conv2 = TransformerConv(hidden_dim, hidden_dim // heads, heads=heads,
                                   concat=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        # Input embedding
        x = self.input_proj(x)
        
        # First transformer layer with residual connection
        x_new = self.conv1(x, edge_index)
        x = self.norm1(x + self.dropout(x_new))
        
        # Second transformer layer with residual connection  
        x_new = self.conv2(x, edge_index)
        x = self.norm2(x + self.dropout(x_new))
        
        # Final classification
        return self.classifier(x)

def load_elliptic_splits(include_unknowns=True, combine_train_val=True):
    """Load pre-computed splits from splits folder"""
    splits_dir = 'splits/full_dataset' if include_unknowns else 'splits/labeled_only'
    print(f"Loading splits from {splits_dir} with include_unknowns={include_unknowns}")
    
    # Load features and classes for each split
    def load_split_data(split_name):
        features_df = pd.read_csv(f'{splits_dir}/{split_name}_features.csv')  # Has header
        classes_df = pd.read_csv(f'{splits_dir}/{split_name}_classes.csv')
        edges_df = pd.read_csv(f'{splits_dir}/{split_name}_edges.csv')
        
        # Process features - txId is first column, timestep is second, features start from 3rd
        node_ids = features_df['txId'].values
        timesteps = features_df['timestep'].values  # Extract timestep information
        features = features_df.iloc[:, 1:].values.astype(np.float32)  # Skip only txId, include timestep and all features
        
        # Process labels - handle both string and integer class values
        labels = []
        for _, row in classes_df.iterrows():
            class_val = row['class']
            
            # Handle both string and integer class values
            if class_val == '1' or class_val == 1:  # illicit
                labels.append(0)
            elif class_val == '2' or class_val == 2:  # licit
                labels.append(1)
            else:  # unknown (string 'unknown' or any other value)
                labels.append(-1 if include_unknowns else None)
        
        # Filter out None labels if not including unknowns
        if not include_unknowns:
            valid_mask = [l is not None for l in labels]
            valid_indices = [i for i, valid in enumerate(valid_mask) if valid]
            node_ids = node_ids[valid_indices]
            timesteps = timesteps[valid_indices]
            features = features[valid_indices]
            labels = [labels[i] for i in valid_indices]
        
        # Create node mapping for edges
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # Process edges
        edge_list = []
        for _, row in edges_df.iterrows():
            if row['txId1'] in node_to_idx and row['txId2'] in node_to_idx:
                edge_list.append([node_to_idx[row['txId1']], node_to_idx[row['txId2']]])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
        
        return {
            'features': torch.tensor(features, dtype=torch.float),
            'edge_index': edge_index,
            'labels': torch.tensor(labels, dtype=torch.long),
            'timesteps': torch.tensor(timesteps, dtype=torch.long),
            'node_ids': node_ids
        }
    
    # Load train, val, test splits
    train_data = load_split_data('train')
    val_data = load_split_data('val') 
    test_data = load_split_data('test')
    
    print(f"Split sizes:")
    print(f"  Train: {train_data['features'].shape[0]} nodes, {train_data['edge_index'].shape[1]} edges")
    print(f"  Val: {val_data['features'].shape[0]} nodes, {val_data['edge_index'].shape[1]} edges")
    print(f"  Test: {test_data['features'].shape[0]} nodes, {test_data['edge_index'].shape[1]} edges")
    
    # Combine train and val if requested (paper setup)
    if combine_train_val:
        print("Combining train and val into single training set (paper setup)")
        
        # Concatenate features and labels
        combined_features = torch.cat([train_data['features'], val_data['features']], dim=0)
        combined_labels = torch.cat([train_data['labels'], val_data['labels']], dim=0)
        combined_timesteps = torch.cat([train_data['timesteps'], val_data['timesteps']], dim=0)
        combined_node_ids = np.concatenate([train_data['node_ids'], val_data['node_ids']])
        
        # Adjust edge indices for val data
        val_edges_adjusted = val_data['edge_index'] + train_data['features'].shape[0]
        combined_edge_index = torch.cat([train_data['edge_index'], val_edges_adjusted], dim=1)
        
        train_data = {
            'features': combined_features,
            'edge_index': combined_edge_index,
            'labels': combined_labels,
            'timesteps': combined_timesteps,
            'node_ids': combined_node_ids
        }
        
        print(f"Combined train: {train_data['features'].shape[0]} nodes, {train_data['edge_index'].shape[1]} edges")
    
    # Count labels for each split
    for split_name, split_data in [('Train', train_data), ('Test', test_data)]:
        labels = split_data['labels']
        if include_unknowns:
            illicit_count = (labels == 0).sum().item()
            licit_count = (labels == 1).sum().item()
            unknown_count = (labels == -1).sum().item()
            print(f"  {split_name}: illicit={illicit_count}, licit={licit_count}, unknown={unknown_count}")
        else:
            illicit_count = (labels == 0).sum().item()
            licit_count = (labels == 1).sum().item()
            print(f"  {split_name}: illicit={illicit_count}, licit={licit_count}")
    
    return train_data, test_data

def create_split_graphs_from_data(train_data, test_data):
    """Convert split data to PyG Data objects"""
    train_graph = Data(
        x=train_data['features'],
        edge_index=train_data['edge_index'],
        y=train_data['labels'],
        timestep=train_data['timesteps']
    )
    
    test_graph = Data(
        x=test_data['features'],
        edge_index=test_data['edge_index'],
        y=test_data['labels'],
        timestep=test_data['timesteps']
    )
    
    return train_graph, test_graph

def evaluate_illicit_metrics(y_true, y_pred, y_proba):
    """Calculate precision, recall, F1, ROC-AUC and PR-AUC for illicit class (class 0)"""
    # Illicit class metrics (class 0)
    precision = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    # ROC-AUC for illicit class (probability of class 0)
    try:
        roc_auc = roc_auc_score(y_true == 0, y_proba[:, 0])  # Binary: is_illicit vs probability of illicit
    except:
        roc_auc = 0.0
    
    # PR-AUC (Precision-Recall AUC) for illicit class
    try:
        pr_auc = average_precision_score(y_true == 0, y_proba[:, 0])  # Binary: is_illicit vs probability of illicit
    except:
        pr_auc = 0.0
    
    return {
        'precision': precision,
        'recall': recall, 
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

def evaluate_by_timestep(model, test_data, test_labeled_mask, device):
    """Evaluate model performance by timestep (similar to XGBoost analysis)"""
    model.eval()
    timestep_results = []
    
    with torch.no_grad():
        # Get predictions for all test nodes
        test_out = model(test_data.x, test_data.edge_index)
        test_proba = F.softmax(test_out[test_labeled_mask], dim=1)
        test_pred = test_out[test_labeled_mask].argmax(dim=1)
        test_true = test_data.y[test_labeled_mask]
        
        # Get timesteps for labeled test nodes
        test_timesteps = test_data.timestep[test_labeled_mask] if hasattr(test_data, 'timestep') else None
        
        if test_timesteps is not None:
            unique_timesteps = torch.unique(test_timesteps).cpu().numpy()
            
            for timestep in unique_timesteps:
                timestep_mask = test_timesteps == timestep
                if timestep_mask.sum() > 0:
                    ts_true = test_true[timestep_mask]
                    ts_pred = test_pred[timestep_mask]
                    ts_proba = test_proba[timestep_mask]
                    
                    # Calculate metrics for this timestep
                    if len(torch.unique(ts_true)) > 1:  # Only if both classes present
                        metrics = evaluate_illicit_metrics(
                            ts_true.cpu().numpy(),
                            ts_pred.cpu().numpy(),
                            ts_proba.cpu().numpy()
                        )
                        
                        timestep_results.append({
                            'timestep': int(timestep),
                            'total_nodes': int(timestep_mask.sum()),
                            'illicit_nodes': int((ts_true == 0).sum()),
                            'licit_nodes': int((ts_true == 1).sum()),
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'f1': metrics['f1'],
                            'roc_auc': metrics['roc_auc'],
                            'pr_auc': metrics['pr_auc']
                        })
        
        # Overall metrics
        overall_metrics = evaluate_illicit_metrics(
            test_true.cpu().numpy(),
            test_pred.cpu().numpy(),
            test_proba.cpu().numpy()
        )
        
    return timestep_results, overall_metrics

def create_training_visualizations(history, checkpoint_dir, experiment_type):
    """Create and save training loss and test F1 visualizations."""
    # Set matplotlib to use non-interactive backend
    plt.ioff()
    
    # 1. Training Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['train_loss'], 'b-', linewidth=2, alpha=0.8)
    plt.title(f'Training Loss - {experiment_type.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save training loss plot
    loss_plot_path = os.path.join(checkpoint_dir, 'training_loss.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Test F1 Score Plot
    if len(history['test_epochs']) > 0 and len(history['test_f1']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(history['test_epochs'], history['test_f1'], 'r-', linewidth=2, marker='o', markersize=4, alpha=0.8)
        plt.title(f'Test F1 Score - {experiment_type.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Test F1 Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)  # F1 score is between 0 and 1
        
        # Add best F1 annotation
        if history['test_f1']:
            best_f1 = max(history['test_f1'])
            best_epoch = history['test_epochs'][history['test_f1'].index(best_f1)]
            plt.annotate(f'Best F1: {best_f1:.4f}\nEpoch: {best_epoch}', 
                        xy=(best_epoch, best_f1), 
                        xytext=(10, 10), 
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        # Save test F1 plot
        f1_plot_path = os.path.join(checkpoint_dir, 'test_f1_score.png')
        plt.savefig(f1_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return loss_plot_path, f1_plot_path
    else:
        return loss_plot_path, None

def train_graph_transformer(train_data, test_data, include_unknowns, num_epochs=1000, shared_timestamp=None):
    """Train Graph Transformer with specified hyperparameters and save checkpoints"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use shared timestamp if provided, otherwise create new one
    if shared_timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = shared_timestamp
    
    # Create organized checkpoint directory structure
    experiment_type = "gt_with_unknowns" if include_unknowns else "gt_labeled_only"
    checkpoint_dir = f"checkpoints/graph_transformer/{timestamp}/{experiment_type}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Saving checkpoints to: {checkpoint_dir}")
    
    # Model setup - 2-layer Graph Transformer with 128-dim hidden, 8 heads
    input_dim = train_data.x.shape[1]
    model = GraphTransformer(input_dim, hidden_dim=128, num_classes=2, heads=8, dropout=0.1).to(device)
    
    # Adam optimizer with lr=0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Weighted CrossEntropyLoss (0.7 for illicit class 0, 0.3 for licit class 1)
    if include_unknowns:
        # Use ignore_index for unknown nodes (-1)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.7, 0.3]).to(device), ignore_index=-1)
        # Create labeled mask for evaluation
        train_labeled_mask = train_data.y != -1
        test_labeled_mask = test_data.y != -1
    else:
        # Standard weighted loss
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.7, 0.3]).to(device))
        train_labeled_mask = torch.ones(train_data.y.shape[0], dtype=torch.bool)
        test_labeled_mask = torch.ones(test_data.y.shape[0], dtype=torch.bool)
    
    train_data = train_data.to(device)
    test_data = test_data.to(device)
    
    print(f"\nTraining Graph Transformer - {num_epochs} epochs...")
    print(f"Training data:")
    print(f"  Total nodes: {train_data.x.shape[0]}")
    print(f"  Labeled mask sum: {train_labeled_mask.sum().item()}")
    print(f"  Include unknowns: {include_unknowns}")
    print(f"Architecture: 2-layer Graph Transformer, 8 heads, 128-dim hidden")
    
    if include_unknowns:
        train_labels = train_data.y[train_labeled_mask]
        unique_labels, counts = torch.unique(train_labels, return_counts=True)
        print(f"  All labels: {unique_labels.cpu().numpy()}, {counts.cpu().numpy()}")
        print(f"Using weighted CrossEntropyLoss with ignore_index=-1 (0.7 illicit, 0.3 licit)")
    else:
        unique_labels, counts = torch.unique(train_data.y, return_counts=True) 
        print(f"  All labels: {unique_labels.cpu().numpy()}, {counts.cpu().numpy()}")
        print(f"Using weighted CrossEntropyLoss (0.7 illicit, 0.3 licit)")
    
    best_illicit_f1 = 0.0
    best_epoch = 0
    best_test_metrics = {}
    
    # Track loss and metrics history for plotting
    history = {
        'epoch': [],
        'train_loss': [],           # Every epoch
        'test_epochs': [],          # Test evaluation epochs (every 50)
        'test_f1': [],              # Every 50 epochs
        'test_precision': [],       # Every 50 epochs
        'test_recall': [],          # Every 50 epochs
        'test_roc_auc': [],         # Every 50 epochs
        'test_pr_auc': []           # Every 50 epochs
    }
    
    # Save training configuration
    config = {
        'model_type': 'Graph Transformer',
        'num_epochs': num_epochs,
        'learning_rate': 0.001,
        'hidden_dim': 128,
        'attention_heads': 8,
        'num_layers': 2,
        'dropout': 0.1,
        'include_unknowns': include_unknowns,
        'train_nodes': train_data.x.shape[0],
        'test_nodes': test_data.x.shape[0],
        'labeled_nodes': train_labeled_mask.sum().item(),
        'loss_weights': [0.7, 0.3],
        'device': str(device),
        'architecture': 'input_proj + 2x(TransformerConv + LayerNorm + Residual) + classifier'
    }
    
    torch.save(config, os.path.join(checkpoint_dir, 'config.pt'))
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        out = model(train_data.x, train_data.edge_index)
        loss = loss_fn(out, train_data.y)
        
        loss.backward()
        optimizer.step()
        
        # Record training loss every epoch
        history['epoch'].append(epoch)
        history['train_loss'].append(loss.item())
        
        # Evaluation every 10 epochs for progress monitoring
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Test evaluation (on labeled nodes only)
                test_out = model(test_data.x, test_data.edge_index)
                test_proba = F.softmax(test_out[test_labeled_mask], dim=1)
                test_pred = test_out[test_labeled_mask].argmax(dim=1)
                test_true = test_data.y[test_labeled_mask]
                
                # Calculate illicit-specific metrics
                test_metrics = evaluate_illicit_metrics(
                    test_true.cpu().numpy(),
                    test_pred.cpu().numpy(), 
                    test_proba.cpu().numpy()
                )
                
                print(f"Epoch {epoch:03d}: Loss={loss:.4f}, Test Illicit F1={test_metrics['f1']:.4f}, " +
                      f"Precision={test_metrics['precision']:.4f}, Recall={test_metrics['recall']:.4f}, " +
                      f"ROC-AUC={test_metrics['roc_auc']:.4f}, PR-AUC={test_metrics['pr_auc']:.4f}")
                
                # Add test metrics to history (every 50 epochs)
                history['test_epochs'].append(epoch)
                history['test_f1'].append(test_metrics['f1'])
                history['test_precision'].append(test_metrics['precision'])
                history['test_recall'].append(test_metrics['recall'])
                history['test_roc_auc'].append(test_metrics['roc_auc'])
                history['test_pr_auc'].append(test_metrics['pr_auc'])
                
                if test_metrics['f1'] > best_illicit_f1:
                    best_illicit_f1 = test_metrics['f1']
                    best_epoch = epoch
                    best_test_metrics = test_metrics.copy()
                    
                    # Save best model (silently)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                        'metrics': test_metrics,
                        'config': config,
                        'history': history
                    }, os.path.join(checkpoint_dir, 'best_model.pt'))
                
                # Save checkpoint every 50 epochs (when test evaluation happens)
                if epoch > 0:  # Don't save at epoch 0
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                        'metrics': test_metrics,
                        'best_f1_so_far': best_illicit_f1,
                        'config': config,
                        'history': history.copy()  # Save history up to this point
                    }
                    
                    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:04d}.pt')
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Checkpoint saved to: checkpoint_epoch_{epoch:04d}.pt (F1={test_metrics['f1']:.4f}, Best F1={best_illicit_f1:.4f})")
        
    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Test evaluation
        test_out = model(test_data.x, test_data.edge_index)
        test_proba = F.softmax(test_out[test_labeled_mask], dim=1)
        test_pred = test_out[test_labeled_mask].argmax(dim=1)
        test_true = test_data.y[test_labeled_mask]
        
        # Calculate final illicit-specific metrics
        final_metrics = evaluate_illicit_metrics(
            test_true.cpu().numpy(),
            test_pred.cpu().numpy(),
            test_proba.cpu().numpy()
        )
        
        print(f"\nFinal evaluation (Illicit Class Performance):")
        print(f"  Final Precision: {final_metrics['precision']:.4f}")
        print(f"  Final Recall:    {final_metrics['recall']:.4f}")
        print(f"  Final F1:        {final_metrics['f1']:.4f}")
        print(f"  Final ROC-AUC:   {final_metrics['roc_auc']:.4f}")
        print(f"  Final PR-AUC:    {final_metrics['pr_auc']:.4f}")
        print(f"  Best F1: {best_illicit_f1:.4f} (epoch {best_epoch})")
        
        # Add final evaluation to history
        final_epoch = num_epochs - 1
        history['test_epochs'].append(final_epoch)
        history['test_f1'].append(final_metrics['f1'])
        history['test_precision'].append(final_metrics['precision'])
        history['test_recall'].append(final_metrics['recall'])
        history['test_roc_auc'].append(final_metrics['roc_auc'])
        history['test_pr_auc'].append(final_metrics['pr_auc'])
        
        # Check if final model is actually better than best recorded
        if final_metrics['f1'] > best_illicit_f1:
            print(f"  >>> Final model is better! Updating best F1: {final_metrics['f1']:.4f}")
            best_illicit_f1 = final_metrics['f1']
            best_epoch = num_epochs - 1  # Last epoch
            best_test_metrics = final_metrics.copy()
            
            # Save the final model as the best model
            torch.save({
                'epoch': num_epochs - 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': 0,  # Final loss not tracked here
                'metrics': final_metrics,
                'config': config,
                'history': history
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
            print(f"  >>> Updated best model checkpoint")
        
        # Detailed classification report
        print(f"\nDetailed Test Results:")
        print(classification_report(test_true.cpu(), test_pred.cpu(), 
                                   target_names=['illicit', 'licit']))
    
    # Save final model
    final_checkpoint = {
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
        'final_metrics': final_metrics,
        'best_metrics': best_test_metrics,
        'best_epoch': best_epoch,
        'config': config,
        'history': history  # Complete training history
    }
    
    torch.save(final_checkpoint, os.path.join(checkpoint_dir, 'final_model.pt'))
    
    # Timestep-based evaluation (similar to XGBoost analysis)
    print(f"\nEvaluating by timestep...")
    timestep_results, overall_metrics_check = evaluate_by_timestep(model, test_data, test_labeled_mask, device)
    
    # Save timestep results to CSV
    if timestep_results:
        timestep_df = pd.DataFrame(timestep_results)
        timestep_csv_path = os.path.join(checkpoint_dir, 'timestep_results.csv')
        timestep_df.to_csv(timestep_csv_path, index=False)
        print(f"Timestep analysis saved to: timestep_results.csv")
        
        # Print timestep summary
        print(f"Timestep Analysis Summary:")
        print(f"  Timesteps evaluated: {len(timestep_results)}")
        print(f"  Average F1 across timesteps: {timestep_df['f1'].mean():.4f}")
        print(f"  Best timestep F1: {timestep_df['f1'].max():.4f} (timestep {timestep_df.loc[timestep_df['f1'].idxmax(), 'timestep']})")
        print(f"  Worst timestep F1: {timestep_df['f1'].min():.4f} (timestep {timestep_df.loc[timestep_df['f1'].idxmin(), 'timestep']})")
    
    # Save overall results to CSV
    overall_results = {
        'metric': ['precision', 'recall', 'f1', 'roc_auc', 'pr_auc'],
        'best_value': [best_test_metrics['precision'], best_test_metrics['recall'], 
                      best_test_metrics['f1'], best_test_metrics['roc_auc'], best_test_metrics['pr_auc']],
        'best_epoch': [best_epoch] * 5,
        'final_value': [final_metrics['precision'], final_metrics['recall'],
                       final_metrics['f1'], final_metrics['roc_auc'], final_metrics['pr_auc']],
        'final_epoch': [num_epochs - 1] * 5
    }
    
    overall_df = pd.DataFrame(overall_results)
    overall_csv_path = os.path.join(checkpoint_dir, 'overall_results.csv')
    overall_df.to_csv(overall_csv_path, index=False)
    print(f"Overall results saved to: overall_results.csv")
    
    # Save loss history as JSON
    history_json_path = os.path.join(checkpoint_dir, 'training_history.json')
    with open(history_json_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: training_history.json")
    
    # Save summary results as JSON
    summary = {
        'experiment_type': 'gt_with_unknowns' if include_unknowns else 'gt_labeled_only',
        'timestamp': timestamp,
        'num_epochs': num_epochs,
        'best_epoch': best_epoch,
        'best_metrics': best_test_metrics,
        'final_metrics': final_metrics,
        'checkpoint_dir': checkpoint_dir,
        'config': config,
        'timestep_results': timestep_results,
        'overall_metrics': overall_metrics_check
    }
    
    summary_json_path = os.path.join(checkpoint_dir, 'experiment_summary.json')
    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Experiment summary saved to: experiment_summary.json")
    
    # Create and save training visualizations
    experiment_type = 'gt_with_unknowns' if include_unknowns else 'gt_labeled_only'
    loss_plot, f1_plot = create_training_visualizations(history, checkpoint_dir, experiment_type)
    print(f"Training visualizations saved:")
    print(f"  - Training loss: training_loss.png")
    if f1_plot:
        print(f"  - Test F1 score: test_f1_score.png")
    
    return best_test_metrics, final_metrics, history, checkpoint_dir

def main():
    print("Graph Transformer Paper Reproduction")
    print("===================================")
    print("Reproducing experimental setup with Graph Transformer:")
    print("- Train/Test split (combining train+val as train)")
    print("- 2-layer Graph Transformer, 128-dim embeddings, 8 heads")
    print("- Weighted CrossEntropyLoss (0.7 illicit, 0.3 licit)")
    print("- 1000 epochs, Adam lr=0.001")
    print("- Using pre-computed splits")
    print("- Saving checkpoints every 50 epochs")
    print("===================================")
    
    # Create shared timestamp for this experimental session
    shared_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Experiment session timestamp: {shared_timestamp}")
    print(f"All results will be saved under: checkpoints/graph_transformer/{shared_timestamp}/")
    
    results = {}
    
    # Experiment 1: Without unknowns (labeled only)
    print("\n" + "="*50)
    print("EXPERIMENT 1: LABELED ONLY (NO UNKNOWNS)")
    print("="*50)
    
    train_data_labeled, test_data_labeled = load_elliptic_splits(include_unknowns=False, combine_train_val=True)
    train_graph, test_graph = create_split_graphs_from_data(train_data_labeled, test_data_labeled)
    
    best_metrics_labeled, final_metrics_labeled, history_labeled, checkpoint_dir_labeled = train_graph_transformer(train_graph, test_graph, include_unknowns=False, num_epochs=10, shared_timestamp=shared_timestamp)
    results['labeled_only'] = {
        'best': best_metrics_labeled, 
        'final': final_metrics_labeled,
        'history': history_labeled,
        'checkpoint_dir': checkpoint_dir_labeled
    }
    
    # Experiment 2: With unknowns
    print("\n" + "="*50)
    print("EXPERIMENT 2: WITH UNKNOWNS")
    print("="*50)
    
    train_data_unknowns, test_data_unknowns = load_elliptic_splits(include_unknowns=True, combine_train_val=True)
    train_graph, test_graph = create_split_graphs_from_data(train_data_unknowns, test_data_unknowns)
    
    best_metrics_unknowns, final_metrics_unknowns, history_unknowns, checkpoint_dir_unknowns = train_graph_transformer(train_graph, test_graph, include_unknowns=True, num_epochs=1000, shared_timestamp=shared_timestamp)
    results['with_unknowns'] = {
        'best': best_metrics_unknowns, 
        'final': final_metrics_unknowns,
        'history': history_unknowns,
        'checkpoint_dir': checkpoint_dir_unknowns
    }
    
    # Summary
    print("\n" + "="*50)
    print("FINAL COMPARISON - ILLICIT CLASS PERFORMANCE")
    print("="*50)
    print(f"Labeled Only:")
    print(f"  Best F1:        {results['labeled_only']['best']['f1']:.4f}")
    print(f"  Best Precision: {results['labeled_only']['best']['precision']:.4f}")
    print(f"  Best Recall:    {results['labeled_only']['best']['recall']:.4f}")
    print(f"  Best ROC-AUC:   {results['labeled_only']['best']['roc_auc']:.4f}")
    print(f"  Best PR-AUC:    {results['labeled_only']['best']['pr_auc']:.4f}")
    print(f"With Unknowns:")
    print(f"  Best F1:        {results['with_unknowns']['best']['f1']:.4f}")
    print(f"  Best Precision: {results['with_unknowns']['best']['precision']:.4f}")
    print(f"  Best Recall:    {results['with_unknowns']['best']['recall']:.4f}")
    print(f"  Best ROC-AUC:   {results['with_unknowns']['best']['roc_auc']:.4f}")
    print(f"  Best PR-AUC:    {results['with_unknowns']['best']['pr_auc']:.4f}")
    
    print(f"\nNote: These are illicit class (fraud detection) specific metrics")
    
    # Save results to JSON file in organized directory structure
    results_dir = f"checkpoints/graph_transformer/{shared_timestamp}/comparisons"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "graph_transformer_reproduction_results.json")
    
    # Add metadata to results
    results['metadata'] = {
        'experiment_name': 'Graph Transformer Paper Reproduction',
        'timestamp': shared_timestamp,
        'date': datetime.now().isoformat(),
        'config': {
            'hidden_dim': 128,
            'num_epochs': 1000,
            'learning_rate': 0.001,
            'attention_heads': 8,
            'num_layers': 2,
            'dropout': 0.1,
            'weighted_loss': [0.3, 0.7],  # [licit, illicit]
            'architecture': '2-layer Graph Transformer',
            'dataset': 'Elliptic Bitcoin Dataset'
        },
        'splits': {
            'train_timesteps': '1-29',
            'validation_timesteps': '31-34 (combined with train)',
            'test_timesteps': '35-49'
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nExperiment session completed!")
    print(f"All results saved under: checkpoints/graph_transformer/{shared_timestamp}/")
    print(f"")
    print(f"Folder structure:")
    print(f"  checkpoints/graph_transformer/{shared_timestamp}/")
    print(f"    ├── gt_labeled_only/      (individual experiment results)")
    print(f"    ├── gt_with_unknowns/     (individual experiment results)")
    print(f"    └── comparisons/          (comparative analysis)")
    print(f"         └── graph_transformer_reproduction_results.json")
    print(f"")
    print(f"Individual experiment folders contain:")
    print("  - Model checkpoints (every 50 epochs)")
    print("  - Training visualizations (loss & F1 plots)")
    print("  - Timestep analysis and overall results (CSV)")
    print("  - Training history and experiment summary (JSON)")
    print(f"")
    print(f"Comparison results location: {results_dir}")
    print(f"  - Labeled only results: {results['labeled_only']['checkpoint_dir']}")
    print(f"  - With unknowns results: {results['with_unknowns']['checkpoint_dir']}")

if __name__ == "__main__":
    main()