"""
Train Classifier for Cluster Galaxy Classification
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
from pathlib import Path


def prepare_features(features_df, exclude_cols=None):
    """
    Prepare feature matrix and labels from features DataFrame.
    
    Parameters
    ----------
    features_df : pandas.DataFrame
        DataFrame with features and 'label' column
    exclude_cols : list
        Columns to exclude from features
        
    Returns
    -------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Labels
    feature_names : list
        List of feature names
    """
    if exclude_cols is None:
        exclude_cols = ['label', 'cluster', 'x', 'y']
    
    # Get feature columns
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    X = features_df[feature_cols].values
    y = features_df['label'].values
    
    # Convert labels to binary (cluster_member = 1, non_member = 0)
    y_binary = (y == 'cluster_member').astype(int)
    
    return X, y_binary, feature_cols


def train_model(X_train, y_train, model_type='random_forest', **kwargs):
    """
    Train a classification model.
    
    Parameters
    ----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training labels
    model_type : str
        'random_forest' or 'svm'
    **kwargs : dict
        Model hyperparameters
        
    Returns
    -------
    model : Trained model
    scaler : StandardScaler (if needed)
    """
    # Scale features (important for SVM, optional for RF)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if model_type == 'random_forest':
        # Default parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'class_weight': 'balanced'  # Handle class imbalance
        }
        params.update(kwargs)
        
        model = RandomForestClassifier(**params)
        model.fit(X_train_scaled, y_train)
        
    elif model_type == 'svm':
        params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'probability': True,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        params.update(kwargs)
        
        model = SVC(**params)
        model.fit(X_train_scaled, y_train)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test, feature_names=None):
    """
    Evaluate model performance.
    
    Parameters
    ----------
    model : Trained model
    scaler : StandardScaler
    X_test : numpy.ndarray
        Test features
    y_test : numpy.ndarray
        Test labels
    feature_names : list
        Feature names for importance analysis
        
    Returns
    -------
    dict : Evaluation metrics
    """
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of cluster_member
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, 
                                   target_names=['non_member', 'cluster_member'],
                                   output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Feature importance (if available)
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
    
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'feature_importance': feature_importance
    }
    
    return results


def train_classifier(features_file, output_dir='models', model_type='random_forest', 
                     test_size=0.2, random_state=42):
    """
    Complete training pipeline.
    
    Parameters
    ----------
    features_file : str
        Path to CSV file with features
    output_dir : str
        Directory to save model
    model_type : str
        'random_forest' or 'svm'
    test_size : float
        Fraction of data for testing
    random_state : int
        Random seed
        
    Returns
    -------
    dict : Training results
    """
    # Load features
    print(f"Loading features from {features_file}...")
    features_df = pd.read_csv(features_file)
    
    print(f"Total samples: {len(features_df)}")
    print(f"Cluster members: {sum(features_df['label'] == 'cluster_member')}")
    print(f"Non-members: {sum(features_df['label'] == 'non_member')}")
    
    # Prepare features
    X, y, feature_names = prepare_features(features_df)
    
    print(f"Number of features: {len(feature_names)}")
    print(f"Features: {feature_names[:5]}...")  # Show first 5
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    print(f"\nTraining {model_type} model...")
    model, scaler = train_model(X_train, y_train, model_type=model_type)
    
    # Evaluate
    print("Evaluating model...")
    results = evaluate_model(model, scaler, X_test, y_test, feature_names)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(f"Accuracy: {results['accuracy']:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, model.predict(scaler.transform(X_test)),
                                target_names=['non_member', 'cluster_member']))
    
    if results['feature_importance']:
        print("\nTop 10 Most Important Features:")
        for i, (feat, importance) in enumerate(list(results['feature_importance'].items())[:10]):
            print(f"  {i+1}. {feat}: {importance:.4f}")
    
    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    model_file = output_path / f'{model_type}_model.pkl'
    scaler_file = output_path / f'{model_type}_scaler.pkl'
    results_file = output_path / f'{model_type}_results.json'
    features_file_saved = output_path / 'feature_names.json'
    
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(features_file_saved, 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    print(f"\n[OK] Model saved to: {model_file}")
    print(f"[OK] Scaler saved to: {scaler_file}")
    print(f"[OK] Results saved to: {results_file}")
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'results': results,
        'model_file': str(model_file),
        'scaler_file': str(scaler_file)
    }
