"""
Main Workflow Script
Complete workflow from data preparation to model training and testing
Supports both FITS and PNG images
"""

import sys
import re
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_preparation import create_unified_dataset
from feature_extraction import extract_features_for_all_galaxies, add_position_features
from train_classifier import train_classifier
from complete_pipeline import run_complete_pipeline
import pandas as pd
import numpy as np


def step1_prepare_data(cluster_names, data_dir, output_dir='processed_data'):
    """
    Step 1: Prepare and organize data.
    
    Parameters
    ----------
    cluster_names : list of str
        List of cluster names
    data_dir : str
        Directory with cluster data
    output_dir : str
        Output directory for processed data
        
    Returns
    -------
    dict : Dataset information
    """
    print("="*60)
    print("STEP 1: DATA PREPARATION")
    print("="*60)
    
    dataset = create_unified_dataset(cluster_names, data_dir, output_dir)
    
    print("\n[OK] Step 1 complete!")
    return dataset


def step2_extract_features(processed_data_dir, output_file='features.csv', 
                          feature_radius=0.01, adaptive_radius=True):
    """
    Step 2: Extract features for all labeled galaxies.
    
    Parameters
    ----------
    processed_data_dir : str
        Directory with processed data
    output_file : str
        Output CSV file for features
    feature_radius : int or float
        Radius for feature extraction. 
        If adaptive_radius=True and < 1, treated as fraction of image size.
        Default 0.01 = 1% of image size (adapts to different pixel scales).
    adaptive_radius : bool
        Adapt radius to image size (recommended when images have different pixel sizes)
        
    Returns
    -------
    str : Path to features file
    """
    print("\n" + "="*60)
    print("STEP 2: FEATURE EXTRACTION")
    print("="*60)
    
    from pathlib import Path
    import numpy as np
    
    processed_path = Path(processed_data_dir)
    all_features = []
    
    # Load unified labels to get cluster info
    unified_labels = pd.read_csv(processed_path / 'unified_labels.csv')
    
    # Process each cluster
    for cluster_name in unified_labels['cluster'].unique():
        print(f"\nProcessing {cluster_name}...")
        
        cluster_dir = processed_path / cluster_name
        if not cluster_dir.exists():
            print(f"  Warning: {cluster_dir} not found, skipping...")
            continue
        
        # Load image
        image_file = cluster_dir / 'image.npy'
        if not image_file.exists():
            print(f"  Warning: {image_file} not found, skipping...")
            continue
        
        image = np.load(image_file)
        
        # Get labels for this cluster
        cluster_labels = unified_labels[unified_labels['cluster'] == cluster_name]
        
        # Extract features (adaptive radius handles different pixel scales)
        features_df = extract_features_for_all_galaxies(
            image, cluster_labels, radius=feature_radius, adaptive_radius=adaptive_radius
        )
        
        # Estimate cluster center (median of all galaxies)
        if len(cluster_labels) > 0:
            center_x = cluster_labels['X'].median()
            center_y = cluster_labels['Y'].median()
            features_df = add_position_features(features_df, center_x, center_y)
        
        all_features.append(features_df)
        print(f"  [OK] Extracted features for {len(features_df)} galaxies")
    
    # Combine all features
    if all_features:
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_features.to_csv(output_file, index=False)
        
        print(f"\n[OK] Extracted features for {len(combined_features)} total galaxies")
        print(f"[OK] Features saved to: {output_file}")
        
        # Print feature summary
        print(f"\nFeature summary:")
        print(f"  Cluster members: {sum(combined_features['label'] == 'cluster_member')}")
        print(f"  Non-members: {sum(combined_features['label'] == 'non_member')}")
        print(f"  Number of features: {len([c for c in combined_features.columns if c not in ['label', 'cluster', 'x', 'y']])}")
        
        return output_file
    else:
        raise ValueError("No features were extracted!")


def step3_train_model(features_file, model_dir='models', model_type='random_forest'):
    """
    Step 3: Train classification model.
    
    Parameters
    ----------
    features_file : str
        Path to features CSV
    model_dir : str
        Directory to save model
    model_type : str
        Model type ('random_forest' or 'svm')
        
    Returns
    -------
    dict : Training results
    """
    print("\n" + "="*60)
    print("STEP 3: MODEL TRAINING")
    print("="*60)
    
    results = train_classifier(
        features_file, 
        output_dir=model_dir,
        model_type=model_type
    )
    
    print("\n[OK] Step 3 complete!")
    return results


def step4_test_model(image_file, model_dir='models', model_type='random_forest',
                     output_dir='results', detection_method='threshold',
                     confidence_threshold=0.7):
    """
    Step 4: Test model on a new image.
    
    Parameters
    ----------
    image_file : str
        Path to FITS or PNG file to test
    model_dir : str
        Directory with trained model
    model_type : str
        Model type
    output_dir : str
        Output directory
    detection_method : str
        Detection method
    confidence_threshold : float
        Classification threshold
        
    Returns
    -------
    dict : Test results
    """
    print("\n" + "="*60)
    print("STEP 4: TESTING MODEL")
    print("="*60)
    
    results = run_complete_pipeline(
        image_file=image_file,
        model_dir=model_dir,
        model_type=model_type,
        detection_method=detection_method,
        output_dir=output_dir,
        confidence_threshold=confidence_threshold
    )
    
    print("\n[OK] Step 4 complete!")
    return results


def run_full_workflow(cluster_names, data_dir, test_image_file=None,
                     processed_data_dir='processed_data',
                     features_file='features.csv',
                     model_dir='models',
                     model_type='random_forest',
                     test_output_dir='results',
                     confidence_threshold=0.7):
    """
    Run complete workflow: prepare → extract → train → test.
    
    Parameters
    ----------
    cluster_names : list of str
        List of cluster names for training
    data_dir : str
        Directory with cluster data
    test_image_file : str, optional
        FITS or PNG file to test on
    processed_data_dir : str
        Directory for processed data
    features_file : str
        Output features file
    model_dir : str
        Model directory
    model_type : str
        Model type
    test_output_dir : str
        Test results directory
    confidence_threshold : float
        Classification confidence threshold (default 0.7)
    """
    print("\n" + "="*60)
    print("CLUSTER GALAXY CLASSIFIER - FULL WORKFLOW")
    print("="*60)
    
    # Step 1: Prepare data
    dataset = step1_prepare_data(cluster_names, data_dir, processed_data_dir)
    
    # Step 2: Extract features
    features_file = step2_extract_features(processed_data_dir, features_file)
    
    # Step 3: Train model
    training_results = step3_train_model(features_file, model_dir, model_type)
    
    # Step 4: Test (if test file provided)
    if test_image_file:
        test_results = step4_test_model(
            test_image_file, model_dir, model_type, test_output_dir,
            confidence_threshold=confidence_threshold
        )
        return training_results, test_results
    else:
        print("\n" + "="*60)
        print("WORKFLOW COMPLETE!")
        print("="*60)
        print(f"\nModel trained and saved to: {model_dir}")
        print(f"Use run_complete_pipeline() to test on new images.")
        return training_results


def discover_cluster_folders(data_dir):
    """
    Automatically discover all cluster folders in the data directory.
    A folder is considered a cluster if it contains:
    - An image file (PNG, JPG, or FITS)
    - A members CSV file
    
    Parameters
    ----------
    data_dir : str
        Directory containing cluster data
        
    Returns
    -------
    list : Sorted list of cluster folder names
    """
    from pathlib import Path
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    cluster_folders = []
    
    # Check each subdirectory
    for item in data_path.iterdir():
        if item.is_dir():
            # Check if it has required files
            has_image = False
            has_members = False
            
            # Check for image files
            image_extensions = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.fits', '.fit', '.FITS']
            for ext in image_extensions:
                if list(item.glob(f'*{ext}')):
                    has_image = True
                    break
            
            # Check for members CSV (various naming patterns)
            members_patterns = [
                f'{item.name}_members.csv',
                'members.csv',
                'cluster_members.csv',
                f'members_{item.name}.csv'
            ]
            for pattern in members_patterns:
                if (item / pattern).exists():
                    has_members = True
                    break
            
            # If it has both image and members file, it's a cluster
            if has_image and has_members:
                cluster_folders.append(item.name)
    
    # Sort clusters naturally (cluster_001, cluster_002, ..., cluster_010, etc.)
    def natural_sort_key(name):
        # Extract numbers from cluster names for natural sorting
        parts = re.split(r'(\d+)', name)
        return [int(part) if part.isdigit() else part.lower() for part in parts]
    
    cluster_folders.sort(key=natural_sort_key)
    
    return cluster_folders


if __name__ == '__main__':
    # ============================================================
    # CONFIGURATION - AUTOMATICALLY FINDS ALL CLUSTERS!
    # ============================================================
    
    # Directory containing your cluster data
    DATA_DIR = 'data'
    
    # Automatically discover all cluster folders
    print("="*60)
    print("DISCOVERING CLUSTER FOLDERS")
    print("="*60)
    CLUSTER_NAMES = discover_cluster_folders(DATA_DIR)
    
    if not CLUSTER_NAMES:
        print("\n[ERROR] No cluster folders found!")
        print(f"Make sure you have folders in {DATA_DIR}/ with:")
        print("  - An image file (PNG, JPG, or FITS)")
        print("  - A members CSV file")
        exit(1)
    
    print(f"\n[OK] Found {len(CLUSTER_NAMES)} cluster(s): {CLUSTER_NAMES}")
    print("\nTo add more clusters:")
    print(f"  1. Create a new folder in {DATA_DIR}/ (e.g., cluster_004)")
    print(f"  2. Add image file and members CSV to that folder")
    print(f"  3. Run this script again - it will automatically find the new cluster!")
    
    # Optional: Test on a new image file after training
    TEST_IMAGE_FILE = None  # e.g., 'data/cluster_004/cluster_004.png'
    
    # Model configuration
    MODEL_TYPE = 'random_forest'  # or 'svm'
    CONFIDENCE_THRESHOLD = 0.7  # Higher = fewer false positives (0.5-0.9)
    
    # ============================================================
    # RUN WORKFLOW
    # ============================================================
    
    print("\n" + "="*60)
    print("STARTING TRAINING WORKFLOW")
    print("="*60)
    
    results = run_full_workflow(
        cluster_names=CLUSTER_NAMES,
        data_dir=DATA_DIR,
        test_image_file=TEST_IMAGE_FILE,
        model_type=MODEL_TYPE,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
    print("\n[OK] All done!")
    print(f"\nTrained on {len(CLUSTER_NAMES)} cluster(s): {', '.join(CLUSTER_NAMES)}")
