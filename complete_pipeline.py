"""
Complete Pipeline: Detect → Extract Features → Classify → Mark → Export
Supports both FITS and PNG images
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import fits
import cv2

from galaxy_detection import detect_galaxies_combined, estimate_cluster_center
from feature_extraction import extract_features_for_all_galaxies, add_position_features


def load_model(model_dir='models', model_type='random_forest'):
    """
    Load trained model and scaler.
    
    Parameters
    ----------
    model_dir : str
        Directory containing saved model
    model_type : str
        Model type ('random_forest' or 'svm')
        
    Returns
    -------
    dict : Dictionary with model, scaler, and feature_names
    """
    model_path = Path(model_dir)
    
    model_file = model_path / f'{model_type}_model.pkl'
    scaler_file = model_path / f'{model_type}_scaler.pkl'
    features_file = model_path / 'feature_names.json'
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    
    with open(features_file, 'r') as f:
        feature_names = json.load(f)
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names
    }


def process_cluster_image(image_file, model_dict, detection_method='comprehensive',
                          detection_params=None, feature_radius=0.01, 
                          confidence_threshold=0.5, adaptive_radius=True):
    """
    Complete pipeline for processing a cluster image.
    Supports both FITS and PNG files.
    
    Parameters
    ----------
    image_file : str
        Path to FITS or PNG file
    model_dict : dict
        Dictionary with model, scaler, feature_names
    detection_method : str
        Detection method ('threshold', 'blob', 'contour', 'all')
    detection_params : dict
        Parameters for detection
    feature_radius : int or float
        Radius for feature extraction (pixels).
        If adaptive_radius=True and < 1, treated as fraction of image size.
        Default 0.01 = 1% of image size (works across different scales).
    confidence_threshold : float
        Minimum confidence to classify as cluster member
    adaptive_radius : bool
        If True, adapts radius to image size (recommended for different pixel scales)
        
    Returns
    -------
    dict : Results including detected galaxies, classifications, and coordinates
    """
    # Load image (FITS or PNG)
    print(f"Loading image: {image_file}")
    image_path = Path(image_file)
    
    if image_path.suffix.lower() in ['.fits', '.fit']:
        # Load FITS
        with fits.open(image_file) as hdul:
            image_data = hdul[0].data
            header = hdul[0].header
        
        # Handle RGB FITS
        if len(image_data.shape) == 3:
            rgb_image = image_data
        elif len(image_data.shape) == 2:
            rgb_image = np.stack([image_data, image_data, image_data], axis=2)
        else:
            raise ValueError(f"Unexpected image shape: {image_data.shape}")
    else:
        # Load PNG, JPG, or other image format
        rgb_image = cv2.imread(str(image_file))
        if rgb_image is None:
            raise FileNotFoundError(f"Could not load image: {image_file}")
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        header = None  # No header for image files
    
    print(f"Image shape: {rgb_image.shape}")
    
    # Step 1: Detect galaxies
    print(f"\nDetecting galaxies using {detection_method} method...")
    if detection_params is None:
        detection_params = {}
    
    detected_galaxies = detect_galaxies_combined(rgb_image, method=detection_method, **detection_params)
    print(f"Detected {len(detected_galaxies)} galaxies")
    
    if len(detected_galaxies) == 0:
        print("Warning: No galaxies detected!")
        return {
            'image': rgb_image,
            'detected_galaxies': [],
            'cluster_members': [],
            'non_members': [],
            'cluster_center': (0, 0)
        }
    
    # Step 2: Estimate cluster center
    cluster_center = estimate_cluster_center(detected_galaxies)
    print(f"Estimated cluster center: ({cluster_center[0]:.1f}, {cluster_center[1]:.1f})")
    
    # Step 3: Extract features
    print("\nExtracting features...")
    galaxies_df = pd.DataFrame(detected_galaxies, columns=['X', 'Y'])
    features_df = extract_features_for_all_galaxies(
        rgb_image, galaxies_df, radius=feature_radius, adaptive_radius=adaptive_radius
    )
    
    # Add position features
    features_df = add_position_features(features_df, cluster_center[0], cluster_center[1])
    
    # Step 4: Classify
    print("Classifying galaxies...")
    model = model_dict['model']
    scaler = model_dict['scaler']
    feature_names = model_dict['feature_names']
    
    # Prepare features (same order as training)
    X = features_df[feature_names].values
    X_scaled = scaler.transform(X)
    
    # Predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]  # Probability of cluster_member
    
    # Add predictions to dataframe
    features_df['prediction'] = predictions
    features_df['probability'] = probabilities
    features_df['is_cluster_member'] = (probabilities >= confidence_threshold)
    
    # Separate cluster members and non-members
    cluster_members = features_df[features_df['is_cluster_member']].copy()
    non_members = features_df[~features_df['is_cluster_member']].copy()
    
    print(f"Classified {len(cluster_members)} as cluster members")
    print(f"Classified {len(non_members)} as non-members")
    
    return {
        'image': rgb_image,
        'detected_galaxies': detected_galaxies,
        'cluster_members': cluster_members,
        'non_members': non_members,
        'cluster_center': cluster_center,
        'features_df': features_df
    }


def visualize_results(results, output_file=None, figsize=(15, 15)):
    """
    Visualize detection and classification results.
    
    Parameters
    ----------
    results : dict
        Results from process_cluster_image
    output_file : str
        Path to save figure
    figsize : tuple
        Figure size
    """
    image = results['image']
    cluster_members = results['cluster_members']
    non_members = results['non_members']
    
    # Convert to grayscale for display if RGB
    if len(image.shape) == 3:
        display_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        display_image = image
    
    # Normalize for display
    vmin = np.percentile(display_image, 1)
    vmax = np.percentile(display_image, 99)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(display_image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    
    # Draw cluster members (red circles)
    if len(cluster_members) > 0:
        for idx, row in cluster_members.iterrows():
            circle = Circle((row['x'], row['y']), 10, 
                          fill=False, color='red', linewidth=2, alpha=0.7)
            ax.add_patch(circle)
    
    # Draw non-members (blue circles, smaller)
    if len(non_members) > 0:
        for idx, row in non_members.iterrows():
            circle = Circle((row['x'], row['y']), 5, 
                          fill=False, color='blue', linewidth=1, alpha=0.5)
            ax.add_patch(circle)
    
    ax.set_title(f'Cluster Galaxy Classification\n'
                f'Red: Cluster Members ({len(cluster_members)}), '
                f'Blue: Non-Members ({len(non_members)})', 
                fontsize=14)
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"[OK] Visualization saved to: {output_file}")
    
    return fig, ax


def export_results(results, output_file='cluster_members.csv'):
    """
    Export cluster member coordinates to CSV.
    
    Parameters
    ----------
    results : dict
        Results from process_cluster_image
    output_file : str
        Output CSV file path
        
    Returns
    -------
    str : Path to saved file
    """
    cluster_members = results['cluster_members']
    
    # Create export dataframe
    export_df = pd.DataFrame({
        'X': cluster_members['x'].values,
        'Y': cluster_members['y'].values,
        'confidence': cluster_members['probability'].values
    })
    
    # Sort by confidence (highest first)
    export_df = export_df.sort_values('confidence', ascending=False)
    
    # Save
    export_df.to_csv(output_file, index=False)
    print(f"[OK] Exported {len(export_df)} cluster members to: {output_file}")
    
    return output_file


def run_complete_pipeline(image_file, model_dir='models', model_type='random_forest',
                         detection_method='comprehensive', output_dir='results',
                         confidence_threshold=0.7, feature_radius=0.01, adaptive_radius=True,
                         detection_params=None):
    """
    Run complete pipeline: detect → classify → visualize → export.
    
    Works on ANY image file size - automatically adapts to different pixel scales.
    Supports both FITS and PNG formats.
    Perfect for testing on new, unseen clusters!
    
    Parameters
    ----------
    image_file : str
        Path to FITS or PNG file (any size, any pixel scale)
    model_dir : str
        Directory with trained model
    model_type : str
        Model type
    detection_method : str
        Detection method
    output_dir : str
        Output directory
    confidence_threshold : float
        Classification confidence threshold (default 0.7 for fewer false positives)
    feature_radius : int or float
        Feature extraction radius. If adaptive_radius=True and < 1, 
        treated as fraction of image size (default 0.01 = 1% of image).
        Use larger values (0.02-0.03) for larger galaxies.
    adaptive_radius : bool
        Adapt radius to image size (recommended for different pixel scales)
        
    Returns
    -------
    dict : Complete results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Set default detection params if not provided (no max_area limit)
    if detection_params is None:
        detection_params = {
            'threshold_percentile': 70,  # Lower threshold to catch brighter/larger objects
            'min_area': 10,  # Increased to reduce small fragments
            'max_area': None,  # No upper limit by default - detects all sizes
            'merge_nearby': True  # Merge nearby detections to reduce over-segmentation
        }
    
    # Load model
    print("Loading trained model...")
    model_dict = load_model(model_dir, model_type)
    
    # Process image
    results = process_cluster_image(
        image_file, model_dict, 
        detection_method=detection_method,
        detection_params=detection_params,
        confidence_threshold=confidence_threshold,
        feature_radius=feature_radius,
        adaptive_radius=adaptive_radius
    )
    
    # Visualize
    image_name = Path(image_file).stem
    viz_file = output_path / f'{image_name}_classified.png'
    visualize_results(results, output_file=str(viz_file))
    
    # Export coordinates
    csv_file = output_path / f'{image_name}_members.csv'
    export_results(results, output_file=str(csv_file))
    
    print(f"\n[OK] Complete! Results saved to: {output_dir}")
    
    return results


if __name__ == '__main__':
    # Example usage
    image_file = 'data/cluster_001/cluster_001.png'  # or .fits
    model_dir = 'models'
    
    results = run_complete_pipeline(
        image_file=image_file,
        model_dir=model_dir,
        detection_method='threshold',
        confidence_threshold=0.7  # Higher threshold for fewer false positives
    )
