"""
Feature Extraction for Galaxy Classification
Extracts color, shape, brightness, and position features from galaxies
"""

import numpy as np
from scipy import ndimage
from skimage import measure, morphology
from skimage.feature import graycomatrix, graycoprops
import cv2
import pandas as pd


def extract_galaxy_features(image, x, y, radius=10, adaptive_radius=True):
    """
    Extract features for a galaxy at position (x, y).
    
    Parameters
    ----------
    image : numpy.ndarray
        RGB image (H, W, 3)
    x, y : float
        Galaxy center coordinates
    radius : int or float
        Radius around galaxy to extract features (pixels).
        If adaptive_radius=True and radius is float, it's treated as fraction of image size.
    adaptive_radius : bool
        If True and radius < 1, treat radius as fraction of image size.
        This helps with different pixel scales.
        
    Returns
    -------
    dict : Dictionary of features
    """
    h, w = image.shape[:2]
    
    # Adaptive radius: if radius < 1, treat as fraction of image size
    if adaptive_radius and isinstance(radius, (float, np.floating)) and radius < 1.0:
        # Use fraction of smaller dimension
        radius = int(min(h, w) * radius)
    elif adaptive_radius and isinstance(radius, (float, np.floating)):
        radius = int(radius)
    else:
        radius = int(radius)
    
    # Convert to integer coordinates
    x_int = int(round(x))
    y_int = int(round(y))
    
    # Ensure coordinates are within image bounds
    x_int = np.clip(x_int, radius, w - radius - 1)
    y_int = np.clip(y_int, radius, h - radius - 1)
    
    # Extract region around galaxy
    y_min = max(0, y_int - radius)
    y_max = min(h, y_int + radius + 1)
    x_min = max(0, x_int - radius)
    x_max = min(w, x_int + radius + 1)
    
    region = image[y_min:y_max, x_min:x_max]
    
    if region.size == 0:
        # Return default features if region is invalid
        return get_default_features()
    
    # Convert RGB to grayscale for some features
    if len(region.shape) == 3:
        gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        r_band = region[:, :, 0]
        g_band = region[:, :, 1]
        b_band = region[:, :, 2]
    else:
        gray_region = region
        r_band = region
        g_band = region
        b_band = region
    
    features = {}
    
    # ========== COLOR FEATURES ==========
    # These are key for distinguishing red cluster members
    
    # Mean colors
    features['mean_r'] = np.mean(r_band)
    features['mean_g'] = np.mean(g_band)
    features['mean_b'] = np.mean(b_band)
    
    # Color ratios (important for red galaxies)
    features['r_g_ratio'] = features['mean_r'] / (features['mean_g'] + 1e-6)
    features['r_b_ratio'] = features['mean_r'] / (features['mean_b'] + 1e-6)
    features['g_b_ratio'] = features['mean_g'] / (features['mean_b'] + 1e-6)
    
    # Color differences
    features['r_g_diff'] = features['mean_r'] - features['mean_g']
    features['r_b_diff'] = features['mean_r'] - features['mean_b']
    
    # Redness index (higher = redder, typical of old cluster galaxies)
    features['redness'] = features['mean_r'] / (features['mean_g'] + features['mean_b'] + 1e-6)
    
    # ========== BRIGHTNESS FEATURES ==========
    features['mean_brightness'] = np.mean(gray_region)
    features['max_brightness'] = np.max(gray_region)
    features['std_brightness'] = np.std(gray_region)
    features['median_brightness'] = np.median(gray_region)
    
    # Brightness at center vs edges
    center_y, center_x = gray_region.shape[0] // 2, gray_region.shape[1] // 2
    center_radius = min(3, gray_region.shape[0] // 4)
    
    # Create mask for center
    y_coords, x_coords = np.ogrid[:gray_region.shape[0], :gray_region.shape[1]]
    center_mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= center_radius**2
    edge_mask = ~center_mask
    
    if np.any(center_mask) and np.any(edge_mask):
        features['center_brightness'] = np.mean(gray_region[center_mask])
        features['edge_brightness'] = np.mean(gray_region[edge_mask])
        features['center_edge_ratio'] = features['center_brightness'] / (features['edge_brightness'] + 1e-6)
    else:
        features['center_brightness'] = features['mean_brightness']
        features['edge_brightness'] = features['mean_brightness']
        features['center_edge_ratio'] = 1.0
    
    # ========== SHAPE FEATURES ==========
    # Threshold to find galaxy shape
    threshold = np.percentile(gray_region, 75)
    binary = gray_region > threshold
    
    # Find largest connected component (the galaxy)
    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)
    
    if len(regions) > 0:
        # Get largest region
        largest_region = max(regions, key=lambda r: r.area)
        
        features['area'] = largest_region.area
        features['perimeter'] = largest_region.perimeter
        features['circularity'] = 4 * np.pi * largest_region.area / (largest_region.perimeter**2 + 1e-6)
        features['eccentricity'] = largest_region.eccentricity
        features['solidity'] = largest_region.solidity
        features['extent'] = largest_region.extent
        
        # Major and minor axis
        features['major_axis'] = largest_region.major_axis_length
        features['minor_axis'] = largest_region.minor_axis_length
        features['aspect_ratio'] = features['major_axis'] / (features['minor_axis'] + 1e-6)
    else:
        # No clear shape detected
        features['area'] = 0
        features['perimeter'] = 0
        features['circularity'] = 0
        features['eccentricity'] = 0
        features['solidity'] = 0
        features['extent'] = 0
        features['major_axis'] = 0
        features['minor_axis'] = 0
        features['aspect_ratio'] = 1.0
    
    # ========== TEXTURE FEATURES ==========
    # Use GLCM (Gray-Level Co-occurrence Matrix) for texture
    try:
        # Normalize to 0-255 for GLCM
        if gray_region.max() > gray_region.min():
            gray_norm = ((gray_region - gray_region.min()) / (gray_region.max() - gray_region.min() + 1e-6) * 255).astype(np.uint8)
        else:
            gray_norm = gray_region.astype(np.uint8)
        
        # Compute GLCM
        glcm = graycomatrix(gray_norm, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        
        # Extract texture properties
        features['contrast'] = graycoprops(glcm, 'contrast')[0, 0]
        features['dissimilarity'] = graycoprops(glcm, 'dissimilarity')[0, 0]
        features['homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
        features['energy'] = graycoprops(glcm, 'energy')[0, 0]
        features['correlation'] = graycoprops(glcm, 'correlation')[0, 0]
    except:
        # If GLCM fails, use default values
        features['contrast'] = 0
        features['dissimilarity'] = 0
        features['homogeneity'] = 1.0
        features['energy'] = 0
        features['correlation'] = 0
    
    # ========== POSITION FEATURES ==========
    # These will be added separately based on cluster center
    # For now, just store raw position
    features['x'] = x
    features['y'] = y
    
    return features


def get_default_features():
    """Return default feature values if extraction fails."""
    return {
        'mean_r': 0, 'mean_g': 0, 'mean_b': 0,
        'r_g_ratio': 1.0, 'r_b_ratio': 1.0, 'g_b_ratio': 1.0,
        'r_g_diff': 0, 'r_b_diff': 0,
        'redness': 1.0,
        'mean_brightness': 0, 'max_brightness': 0, 'std_brightness': 0,
        'median_brightness': 0,
        'center_brightness': 0, 'edge_brightness': 0, 'center_edge_ratio': 1.0,
        'area': 0, 'perimeter': 0, 'circularity': 0,
        'eccentricity': 0, 'solidity': 0, 'extent': 0,
        'major_axis': 0, 'minor_axis': 0, 'aspect_ratio': 1.0,
        'contrast': 0, 'dissimilarity': 0, 'homogeneity': 1.0,
        'energy': 0, 'correlation': 0,
        'x': 0, 'y': 0
    }


def extract_features_for_all_galaxies(image, coordinates_df, radius=10, adaptive_radius=True):
    """
    Extract features for all galaxies in a DataFrame.
    
    Parameters
    ----------
    image : numpy.ndarray
        RGB image
    coordinates_df : pandas.DataFrame
        DataFrame with 'X' and 'Y' columns
    radius : int or float
        Extraction radius. If adaptive_radius=True and < 1, treated as fraction of image size.
    adaptive_radius : bool
        Enable adaptive radius for different image scales
        
    Returns
    -------
    pandas.DataFrame : DataFrame with features for each galaxy
    """
    features_list = []
    
    for idx, row in coordinates_df.iterrows():
        x, y = row['X'], row['Y']
        features = extract_galaxy_features(image, x, y, radius=radius, adaptive_radius=adaptive_radius)
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    # Add labels if present
    if 'label' in coordinates_df.columns:
        features_df['label'] = coordinates_df['label'].values
    if 'cluster' in coordinates_df.columns:
        features_df['cluster'] = coordinates_df['cluster'].values
    
    return features_df


def add_position_features(features_df, cluster_center_x, cluster_center_y):
    """
    Add position-based features relative to cluster center.
    
    Parameters
    ----------
    features_df : pandas.DataFrame
        Features dataframe with 'x' and 'y' columns
    cluster_center_x, cluster_center_y : float
        Cluster center coordinates
        
    Returns
    -------
    pandas.DataFrame : Features with added position features
    """
    features_df = features_df.copy()
    
    # Distance from cluster center
    features_df['distance_from_center'] = np.sqrt(
        (features_df['x'] - cluster_center_x)**2 + 
        (features_df['y'] - cluster_center_y)**2
    )
    
    # Normalized distance (0-1 scale)
    max_dist = features_df['distance_from_center'].max()
    if max_dist > 0:
        features_df['normalized_distance'] = features_df['distance_from_center'] / max_dist
    else:
        features_df['normalized_distance'] = 0
    
    return features_df
