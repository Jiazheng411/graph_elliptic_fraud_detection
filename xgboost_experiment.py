"""
XGBoost Bitcoin Fraud Detection
Clean script for training and evaluating XGBoost on the Elliptic dataset.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score
)
import os

def evaluate_by_timestep(model, test_features, test_classes, use_timestep=False, normalize=False, scaler=None):
    """
    Evaluate model performance by timestep.
    
    Args:
        model: Trained XGBoost model
        test_features: Test features dataframe
        test_classes: Test classes dataframe  
        use_timestep: Whether timestep was used as feature
        normalize: Whether features were normalized
        scaler: Fitted scaler if normalization was used
    
    Returns:
        DataFrame with results by timestep
    """
    # Prepare feature columns
    if use_timestep:
        feature_cols = list(test_features.columns)[1:]  # Include timestep
    else:
        feature_cols = list(test_features.columns)[2:]  # Exclude timestep
    
    # Merge features and classes
    test_data = test_features.merge(test_classes, on='txId')
    
    # Get unique timesteps
    timesteps = sorted(test_data['timestep'].unique())
    
    results = []
    
    # Overall results
    X_all = test_data[feature_cols].values
    y_all = test_data['class'].map({1: 1, 2: 0, '1': 1, '2': 0}).values
    
    if normalize and scaler is not None:
        X_all = scaler.transform(X_all)
    
    y_pred_all = model.predict(X_all)
    y_proba_all = model.predict_proba(X_all)[:, 1]
    
    # Overall metrics
    roc_auc_all = roc_auc_score(y_all, y_proba_all)
    pr_auc_all = average_precision_score(y_all, y_proba_all)
    report_all = classification_report(y_all, y_pred_all, output_dict=True, zero_division=0)
    fraud_metrics_all = report_all.get('1', {})
    
    results.append({
        'timestep': 'ALL',
        'num_transactions': len(y_all),
        'num_fraud': np.sum(y_all),
        'fraud_rate': np.mean(y_all),
        'roc_auc': roc_auc_all,
        'pr_auc': pr_auc_all,
        'fraud_precision': fraud_metrics_all.get('precision', 0.0),
        'fraud_recall': fraud_metrics_all.get('recall', 0.0),
        'fraud_f1': fraud_metrics_all.get('f1-score', 0.0)
    })
    
    # By timestep
    for timestep in timesteps:
        timestep_data = test_data[test_data['timestep'] == timestep]
        
        if len(timestep_data) == 0:
            continue
            
        X_timestep = timestep_data[feature_cols].values
        y_timestep = timestep_data['class'].map({1: 1, 2: 0, '1': 1, '2': 0}).values
        
        if normalize and scaler is not None:
            X_timestep = scaler.transform(X_timestep)
        
        # Skip if no fraud cases in this timestep
        if np.sum(y_timestep) == 0:
            results.append({
                'timestep': timestep,
                'num_transactions': len(y_timestep),
                'num_fraud': 0,
                'fraud_rate': 0.0,
                'roc_auc': np.nan,
                'pr_auc': np.nan,
                'fraud_precision': np.nan,
                'fraud_recall': np.nan,
                'fraud_f1': np.nan
            })
            continue
        
        y_pred_timestep = model.predict(X_timestep)
        y_proba_timestep = model.predict_proba(X_timestep)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_timestep, y_proba_timestep)
        pr_auc = average_precision_score(y_timestep, y_proba_timestep)
        report = classification_report(y_timestep, y_pred_timestep, output_dict=True, zero_division=0)
        fraud_metrics = report.get('1', {})
        
        results.append({
            'timestep': timestep,
            'num_transactions': len(y_timestep),
            'num_fraud': np.sum(y_timestep),
            'fraud_rate': np.mean(y_timestep),
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'fraud_precision': fraud_metrics.get('precision', 0.0),
            'fraud_recall': fraud_metrics.get('recall', 0.0),
            'fraud_f1': fraud_metrics.get('f1-score', 0.0)
        })
    
    return pd.DataFrame(results)

def run_xgboost_experiment(use_timestep=False, normalize=False, combine_train_val=False):
    """
    Run XGBoost experiment for fraud detection.
    
    Args:
        use_timestep: Whether to include timestep as a feature
        normalize: Whether to normalize features
        combine_train_val: Whether to combine train and val sets for training (paper setup)
    
    Returns:
        Tuple of (trained model, scaler if used, test results by timestep)
    """
    print(f"XGBoost Fraud Detection Experiment")
    setup_desc = f"Configuration: {('Timestep + ' if use_timestep else '') + ('Normalized' if normalize else 'Raw')}"
    if combine_train_val:
        setup_desc += " + Combined Train+Val"
    print(setup_desc)
    print("-" * 50)
    
    # Load data
    data_dir = "splits/labeled_only"
    train_features = pd.read_csv(os.path.join(data_dir, 'train_features.csv'))
    train_classes = pd.read_csv(os.path.join(data_dir, 'train_classes.csv'))
    val_features = pd.read_csv(os.path.join(data_dir, 'val_features.csv'))
    val_classes = pd.read_csv(os.path.join(data_dir, 'val_classes.csv'))
    test_features = pd.read_csv(os.path.join(data_dir, 'test_features.csv'))
    test_classes = pd.read_csv(os.path.join(data_dir, 'test_classes.csv'))
    
    print(f"Data loaded: Train={len(train_features):,}, Val={len(val_features):,}, Test={len(test_features):,}")
    
    # Combine train and val if requested (paper setup)
    if combine_train_val:
        print("Combining train and validation sets for training (paper reproduction setup)")
        train_features = pd.concat([train_features, val_features], ignore_index=True)
        train_classes = pd.concat([train_classes, val_classes], ignore_index=True)
        print(f"Combined training data: {len(train_features):,} samples")
        
        # For evaluation, we'll only show train and test (no separate val)
        eval_sets = [("Train", train_features, train_classes), ("Test", test_features, test_classes)]
    else:
        # Standard train/val/test split
        eval_sets = [("Train", train_features, train_classes), ("Val", val_features, val_classes), ("Test", test_features, test_classes)]
    
    # Prepare features
    if use_timestep:
        feature_cols = list(train_features.columns)[1:]  # Include timestep
    else:
        feature_cols = list(train_features.columns)[2:]  # Exclude timestep
    
    X_train = train_features[feature_cols].values
    X_test = test_features[feature_cols].values
    
    # Prepare labels (1=fraud, 0=licit)
    y_train = train_classes['class'].map({1: 1, 2: 0, '1': 1, '2': 0}).values
    y_test = test_classes['class'].map({1: 1, 2: 0, '1': 1, '2': 0}).values
    
    # Prepare validation data if not combining train+val
    if not combine_train_val:
        X_val = val_features[feature_cols].values
        y_val = val_classes['class'].map({1: 1, 2: 0, '1': 1, '2': 0}).values
    
    # Normalize if requested
    scaler = None
    if normalize:
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        if not combine_train_val:
            X_val = scaler.transform(X_val)
        print("Features normalized using RobustScaler")
    
    print(f"Features: {len(feature_cols)}")
    if combine_train_val:
        print(f"Fraud rate - Train+Val: {np.mean(y_train)*100:.1f}%, Test: {np.mean(y_test)*100:.1f}%")
    else:
        print(f"Fraud rate - Train: {np.mean(y_train)*100:.1f}%, Val: {np.mean(y_val)*100:.1f}%, Test: {np.mean(y_test)*100:.1f}%")
    
    # Train model
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    
    print(f"Training XGBoost (scale_pos_weight={scale_pos_weight:.1f})...")
    model.fit(X_train, y_train)
    print("Training completed")
    
    # Evaluate
    print(f"\nFraud Detection Results:")
    print("-" * 30)
    
    for name, features_df, classes_df in eval_sets:
        X = features_df[feature_cols].values
        y = classes_df['class'].map({1: 1, 2: 0, '1': 1, '2': 0}).values
        
        if normalize and scaler is not None:
            X = scaler.transform(X)
        
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        # Core metrics
        roc_auc = roc_auc_score(y, y_proba)
        pr_auc = average_precision_score(y, y_proba)
        
        # Fraud class metrics
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        fraud_precision = report.get('1', {}).get('precision', 0.0)
        fraud_recall = report.get('1', {}).get('recall', 0.0)
        fraud_f1 = report.get('1', {}).get('f1-score', 0.0)
        
        print(f"{name:5} - ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
        print(f"      - Fraud Precision: {fraud_precision:.4f}, Recall: {fraud_recall:.4f}, F1: {fraud_f1:.4f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Features:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['feature']:15s}: {row['importance']:.4f}")
    
    # Analyze by timestep
    print(f"\nAnalyzing performance by timestep...")
    timestep_results = evaluate_by_timestep(model, test_features, test_classes, use_timestep, normalize, scaler)
    
    return model, scaler, timestep_results

def compare_configurations():
    """Compare different XGBoost configurations."""
    print("Comparing XGBoost Configurations")
    print("=" * 50)
    
    configs = [
        {"use_timestep": False, "normalize": False, "name": "Raw Features"},
        {"use_timestep": False, "normalize": True, "name": "Normalized"},
        {"use_timestep": True, "normalize": False, "name": "Raw + Timestep"},
        {"use_timestep": True, "normalize": True, "name": "Normalized + Timestep"}
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        print("-" * 30)
        
        try:
            model = run_xgboost_experiment(
                use_timestep=config['use_timestep'],
                normalize=config['normalize']
            )
            
            # Get test performance
            test_features = pd.read_csv("splits/labeled_only/test_features.csv")
            test_classes = pd.read_csv("splits/labeled_only/test_classes.csv")
            
            if config['use_timestep']:
                feature_cols = list(test_features.columns)[1:]
            else:
                feature_cols = list(test_features.columns)[2:]
            
            X_test = test_features[feature_cols].values
            y_test = test_classes['class'].map({1: 1, 2: 0, '1': 1, '2': 0}).values
            
            if config['normalize']:
                # Note: In practice, save/load the scaler
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                train_features = pd.read_csv("splits/labeled_only/train_features.csv")
                X_train_temp = train_features[feature_cols].values
                scaler.fit(X_train_temp)
                X_test = scaler.transform(X_test)
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
            
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            fraud_precision = report.get('1', {}).get('precision', 0.0)
            fraud_recall = report.get('1', {}).get('recall', 0.0)
            fraud_f1 = report.get('1', {}).get('f1-score', 0.0)
            
            results.append({
                'config': config['name'],
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'fraud_precision': fraud_precision,
                'fraud_recall': fraud_recall,
                'fraud_f1': fraud_f1
            })
            
        except Exception as e:
            print(f"Configuration failed: {e}")
    
    # Summary
    print(f"\nComparison Summary:")
    print("-" * 70)
    print(f"{'Configuration':<20} {'ROC-AUC':<10} {'PR-AUC':<10} {'F-Prec':<8} {'F-Rec':<8} {'F-F1':<8}")
    print("-" * 70)
    
    for result in sorted(results, key=lambda x: x['roc_auc'], reverse=True):
        print(f"{result['config']:<20} {result['roc_auc']:<10.4f} {result['pr_auc']:<10.4f} "
              f"{result['fraud_precision']:<8.4f} {result['fraud_recall']:<8.4f} {result['fraud_f1']:<8.4f}")
    
    if results:
        best = max(results, key=lambda x: x['roc_auc'])
        print(f"\nBest configuration: {best['config']}")
        print(f"ROC-AUC: {best['roc_auc']:.4f}, Fraud F1: {best['fraud_f1']:.4f}")

def run_both_experiments():
    """Run XGBoost experiments both with and without timestep."""
    print("Running XGBoost Experiments: With and Without Timestep")
    print("=" * 60)
    
    # Experiment 1: Without timestep
    print("\nEXPERIMENT 1: WITHOUT TIMESTEP")
    print("=" * 40)
    model1, scaler1, timestep_results1 = run_xgboost_experiment(use_timestep=False, normalize=False)
    
    # Save timestep results
    timestep_results1['experiment'] = 'without_timestep'
    timestep_results1.to_csv('xgboost_results_without_timestep.csv', index=False)
    print(f"Timestep analysis saved to: xgboost_results_without_timestep.csv")
    
    print("\nEXPERIMENT 2: WITH TIMESTEP")
    print("=" * 40)
    model2, scaler2, timestep_results2 = run_xgboost_experiment(use_timestep=True, normalize=False)
    
    # Save timestep results
    timestep_results2['experiment'] = 'with_timestep'
    timestep_results2.to_csv('xgboost_results_with_timestep.csv', index=False)
    print(f"Timestep analysis saved to: xgboost_results_with_timestep.csv")
    
    # Combine results for comparison
    combined_results = pd.concat([timestep_results1, timestep_results2], ignore_index=True)
    combined_results.to_csv('xgboost_combined_results.csv', index=False)
    print(f"Combined results saved to: xgboost_combined_results.csv")
    
    # Print comparison summary
    print(f"\nCOMPARISON SUMMARY")
    print("=" * 40)
    
    # Overall results comparison
    overall1 = timestep_results1[timestep_results1['timestep'] == 'ALL'].iloc[0]
    overall2 = timestep_results2[timestep_results2['timestep'] == 'ALL'].iloc[0]
    
    print(f"Overall Test Results:")
    print(f"                     Without Timestep    With Timestep")
    print(f"ROC-AUC:             {overall1['roc_auc']:13.4f}    {overall2['roc_auc']:12.4f}")
    print(f"PR-AUC:              {overall1['pr_auc']:13.4f}    {overall2['pr_auc']:12.4f}")
    print(f"Fraud Precision:     {overall1['fraud_precision']:13.4f}    {overall2['fraud_precision']:12.4f}")
    print(f"Fraud Recall:        {overall1['fraud_recall']:13.4f}    {overall2['fraud_recall']:12.4f}")
    print(f"Fraud F1:            {overall1['fraud_f1']:13.4f}    {overall2['fraud_f1']:12.4f}")
    
    # Show some timestep-level results
    print(f"\nSample Timestep Results (timesteps 35-39):")
    print(f"Timestep   Without_ROC   With_ROC   Without_F1   With_F1")
    print("-" * 55)
    
    for ts in [35, 36, 37, 38, 39]:
        row1 = timestep_results1[timestep_results1['timestep'] == ts]
        row2 = timestep_results2[timestep_results2['timestep'] == ts]
        
        if len(row1) > 0 and len(row2) > 0:
            roc1 = row1['roc_auc'].iloc[0] if not pd.isna(row1['roc_auc'].iloc[0]) else 0.0
            roc2 = row2['roc_auc'].iloc[0] if not pd.isna(row2['roc_auc'].iloc[0]) else 0.0
            f1_1 = row1['fraud_f1'].iloc[0] if not pd.isna(row1['fraud_f1'].iloc[0]) else 0.0
            f1_2 = row2['fraud_f1'].iloc[0] if not pd.isna(row2['fraud_f1'].iloc[0]) else 0.0
            
            print(f"{ts:8d}   {roc1:9.4f}   {roc2:8.4f}   {f1_1:10.4f}   {f1_2:7.4f}")
    
    return model1, model2, combined_results

def run_both_experiments_combined_trainval():
    """Run XGBoost experiments both with and without timestep using combined train+val."""
    print("Running XGBoost Experiments: With and Without Timestep (Combined Train+Val)")
    print("=" * 70)
    
    # Experiment 1: Without timestep (combined train+val)
    print("\nEXPERIMENT 1: WITHOUT TIMESTEP (COMBINED TRAIN+VAL)")
    print("=" * 50)
    model1, scaler1, timestep_results1 = run_xgboost_experiment(
        use_timestep=False, 
        normalize=False, 
        combine_train_val=True
    )
    
    # Save timestep results
    timestep_results1['experiment'] = 'without_timestep_combined'
    timestep_results1.to_csv('xgboost_results_without_timestep.csv', index=False)
    print(f"Results saved to: xgboost_results_without_timestep.csv")
    
    print("\nEXPERIMENT 2: WITH TIMESTEP (COMBINED TRAIN+VAL)")
    print("=" * 50)
    model2, scaler2, timestep_results2 = run_xgboost_experiment(
        use_timestep=True, 
        normalize=False, 
        combine_train_val=True
    )
    
    # Save timestep results
    timestep_results2['experiment'] = 'with_timestep_combined'
    timestep_results2.to_csv('xgboost_results_with_timestep.csv', index=False)
    print(f"Results saved to: xgboost_results_with_timestep.csv")
    
    # Print comparison summary
    print(f"\nCOMPARISON SUMMARY (COMBINED TRAIN+VAL)")
    print("=" * 50)
    
    # Overall results comparison
    overall1 = timestep_results1[timestep_results1['timestep'] == 'ALL'].iloc[0]
    overall2 = timestep_results2[timestep_results2['timestep'] == 'ALL'].iloc[0]
    
    print(f"Overall Test Results:")
    print(f"                     Without Timestep    With Timestep")
    print(f"ROC-AUC:             {overall1['roc_auc']:13.4f}    {overall2['roc_auc']:12.4f}")
    print(f"PR-AUC:              {overall1['pr_auc']:13.4f}    {overall2['pr_auc']:12.4f}")
    print(f"Fraud Precision:     {overall1['fraud_precision']:13.4f}    {overall2['fraud_precision']:12.4f}")
    print(f"Fraud Recall:        {overall1['fraud_recall']:13.4f}    {overall2['fraud_recall']:12.4f}")
    print(f"Fraud F1:            {overall1['fraud_f1']:13.4f}    {overall2['fraud_f1']:12.4f}")
    
    # Determine winner
    if overall1['fraud_f1'] > overall2['fraud_f1']:
        print(f"\nBest Configuration: Without Timestep (F1={overall1['fraud_f1']:.4f})")
    else:
        print(f"\nBest Configuration: With Timestep (F1={overall2['fraud_f1']:.4f})")
    
    return model1, model2, (timestep_results1, timestep_results2)

if __name__ == "__main__":
    # Run both experiments with combined train+val
    model_without, model_with, (results_without, results_with) = run_both_experiments_combined_trainval()
    
    print(f"\nExperiments completed!")
    print(f"Results saved to:")
    print(f"  - xgboost_results_without_timestep.csv")
    print(f"  - xgboost_results_with_timestep.csv")
    print(f"Both experiments use combined train+val for paper reproduction consistency.")