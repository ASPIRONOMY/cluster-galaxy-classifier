"""
Galaxy Detection for Extended Objects
Detects galaxies in images using various CV techniques
Supports both FITS and PNG images
"""

import numpy as np
from scipy import ndimage
from skimage import measure, morphology, filters
from skimage.feature import blob_log
import cv2


def detect_galaxies_threshold(image, threshold_percentile=85, min_area=5, max_area=None, merge_nearby=True):
    """
    Detect galaxies using thresholding and connected components.
    Good for extended objects.
    
    Parameters
    ----------
    image : numpy.ndarray
        Grayscale or RGB image
    threshold_percentile : float
        Percentile for threshold (0-100). Lower values catch brighter objects.
    min_area : int
        Minimum area in pixels for a detected object
    max_area : int or None
        Maximum area in pixels for a detected object. If None, no upper limit.
    merge_nearby : bool
        If True, merge nearby detections to reduce over-segmentation (default True)
        
    Returns
    -------
    list of tuples : [(x, y), ...] coordinates of detected galaxies
    """
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Normalize to 0-255
    if gray.max() > 255:
        gray = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)
    else:
        gray = gray.astype(np.uint8)
    
    # Try multiple thresholds to catch both bright and dim objects
    # Lower threshold for bright objects, higher for dim objects
    thresholds = [
        np.percentile(gray, threshold_percentile - 10),  # Lower threshold for bright objects
        np.percentile(gray, threshold_percentile),        # Standard threshold
    ]
    
    all_galaxies = []
    
    for threshold in thresholds:
        binary = gray > threshold
        
        # Remove small objects (noise)
        binary = morphology.remove_small_objects(binary, min_size=min_area)
        
        # Find connected components
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)
        
        # Extract centroids
        for region in regions:
            # Check min_area, and max_area only if specified
            if region.area >= min_area:
                if max_area is None or region.area <= max_area:
                    # Get centroid (note: skimage uses (row, col) = (y, x)
                    y, x = region.centroid
                    all_galaxies.append((x, y))
    
    # Remove duplicates and merge nearby detections
    if merge_nearby:
        # Use conservative merge distance (12 pixels) to only merge obvious fragments
        galaxies = merge_nearby_detections(all_galaxies, merge_distance=12)
    else:
        # Just remove exact/close duplicates (within 5 pixels)
        unique_galaxies = []
        for x, y in all_galaxies:
            is_duplicate = False
            for ux, uy in unique_galaxies:
                if np.sqrt((x - ux)**2 + (y - uy)**2) < 5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_galaxies.append((x, y))
        galaxies = unique_galaxies
    
    return galaxies


def detect_large_bright_objects(image, min_area=100, brightness_percentile=90):
    """
    Specifically detect large, bright objects that might be missed by standard thresholding.
    Uses lower threshold to catch very bright objects.
    
    Parameters
    ----------
    image : numpy.ndarray
        Grayscale or RGB image
    min_area : int
        Minimum area for large objects (default 100 pixels)
    brightness_percentile : float
        Percentile for brightness threshold (default 90 = top 10% brightest)
        
    Returns
    -------
    list of tuples : [(x, y), ...] coordinates of detected large bright objects
    """
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Normalize to 0-255
    if gray.max() > 255:
        gray = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)
    else:
        gray = gray.astype(np.uint8)
    
    # Use lower threshold to catch bright objects (top 10-20% brightest)
    threshold = np.percentile(gray, brightness_percentile)
    binary = gray > threshold
    
    # Find connected components
    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)
    
    # Extract centroids of large objects
    galaxies = []
    for region in regions:
        if region.area >= min_area:  # Only large objects
            y, x = region.centroid
            galaxies.append((x, y))
    
    return galaxies


def merge_nearby_detections(galaxies, merge_distance=12):
    """
    Merge nearby detections that are likely parts of the same object.
    This helps reduce over-segmentation.
    
    Parameters
    ----------
    galaxies : list of tuples
        List of (x, y) coordinates
    merge_distance : float
        Maximum distance to merge detections (default 12 pixels - conservative to avoid merging separate objects)
        
    Returns
    -------
    list of tuples : Merged galaxy coordinates
    """
    if len(galaxies) == 0:
        return []
    
    # First, remove exact/very close duplicates (within 5 pixels) from multi-threshold
    unique_galaxies = []
    for x, y in galaxies:
        is_duplicate = False
        for ux, uy in unique_galaxies:
            if np.sqrt((x - ux)**2 + (y - uy)**2) < 5:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_galaxies.append((x, y))
    
    # Now merge only very close ones (likely fragments of same object)
    # Use smaller distance to be conservative
    galaxies_array = np.array(unique_galaxies)
    merged = []
    used = set()
    
    for i, (x, y) in enumerate(unique_galaxies):
        if i in used:
            continue
        
        # Find all nearby detections
        nearby_indices = [i]
        for j, (ux, uy) in enumerate(unique_galaxies):
            if j != i and j not in used:
                dist = np.sqrt((x - ux)**2 + (y - uy)**2)
                if dist < merge_distance:
                    nearby_indices.append(j)
        
        # Only merge if we found nearby ones (otherwise keep original)
        if len(nearby_indices) > 1:
            # Calculate centroid of merged group
            group_coords = galaxies_array[nearby_indices]
            merged_x = np.mean(group_coords[:, 0])
            merged_y = np.mean(group_coords[:, 1])
            merged.append((merged_x, merged_y))
            # Mark all in group as used
            used.update(nearby_indices)
        else:
            # Keep original if no nearby detections
            merged.append((x, y))
            used.add(i)
    
    return merged


def detect_galaxies_blob(image, min_sigma=2, max_sigma=10, num_sigma=10, threshold=0.1):
    """
    Detect galaxies using Laplacian of Gaussian (LoG) blob detection.
    Good for finding circular/elliptical objects.
    
    Parameters
    ----------
    image : numpy.ndarray
        Grayscale or RGB image
    min_sigma, max_sigma : float
        Range of sigma values for blob detection
    num_sigma : int
        Number of sigma values to try
    threshold : float
        Detection threshold
        
    Returns
    -------
    list of tuples : [(x, y), ...] coordinates of detected galaxies
    """
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Normalize
    if gray.max() > 255:
        gray = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)
    else:
        gray = gray.astype(np.uint8)
    
    # Detect blobs
    blobs = blob_log(gray, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    
    # Extract coordinates (blob_log returns [y, x, sigma])
    galaxies = [(blob[1], blob[0]) for blob in blobs]  # (x, y)
    
    return galaxies


def detect_galaxies_contour(image, threshold_percentile=80, min_area=10):
    """
    Detect galaxies using contour detection.
    Good for irregular/extended objects.
    
    Parameters
    ----------
    image : numpy.ndarray
        Grayscale or RGB image
    threshold_percentile : float
        Percentile for threshold
    min_area : int
        Minimum contour area
        
    Returns
    -------
    list of tuples : [(x, y), ...] coordinates of detected galaxies
    """
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Normalize
    if gray.max() > 255:
        gray = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)
    else:
        gray = gray.astype(np.uint8)
    
    # Apply threshold
    threshold = np.percentile(gray, threshold_percentile)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract centroids
    galaxies = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                galaxies.append((x, y))
    
    return galaxies


def detect_galaxies_combined(image, method='threshold', **kwargs):
    """
    Combined detection function that tries multiple methods.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
    method : str
        'threshold', 'blob', 'contour', 'all', or 'comprehensive'
    **kwargs : dict
        Additional parameters for detection methods
        
    Returns
    -------
    list of tuples : Detected galaxy coordinates
    """
    if method == 'threshold':
        return detect_galaxies_threshold(image, **kwargs)
    elif method == 'blob':
        return detect_galaxies_blob(image, **kwargs)
    elif method == 'contour':
        return detect_galaxies_contour(image, **kwargs)
    elif method == 'all':
        # Combine all methods and remove duplicates
        all_galaxies = []
        
        # Filter kwargs for each method
        threshold_kwargs = {k: v for k, v in kwargs.items() 
                           if k in ['threshold_percentile', 'min_area', 'max_area']}
        blob_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['min_sigma', 'max_sigma', 'num_sigma', 'threshold']}
        contour_kwargs = {k: v for k, v in kwargs.items() 
                         if k in ['threshold_percentile', 'min_area']}
        
        all_galaxies.extend(detect_galaxies_threshold(image, **threshold_kwargs))
        all_galaxies.extend(detect_galaxies_blob(image, **blob_kwargs))
        all_galaxies.extend(detect_galaxies_contour(image, **contour_kwargs))
        
        # Remove duplicates (within 5 pixels)
        unique_galaxies = []
        for x, y in all_galaxies:
            is_duplicate = False
            for ux, uy in unique_galaxies:
                if np.sqrt((x - ux)**2 + (y - uy)**2) < 5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_galaxies.append((x, y))
        
        return unique_galaxies
    elif method == 'comprehensive':
        # Comprehensive method: combines all methods + specifically looks for large bright objects
        all_galaxies = []
        
        # Standard methods - filter kwargs for each method
        threshold_kwargs = {k: v for k, v in kwargs.items() 
                           if k in ['threshold_percentile', 'min_area', 'max_area']}
        blob_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['min_sigma', 'max_sigma', 'num_sigma', 'threshold']}
        contour_kwargs = {k: v for k, v in kwargs.items() 
                         if k in ['threshold_percentile', 'min_area']}
        
        all_galaxies.extend(detect_galaxies_threshold(image, **threshold_kwargs))
        all_galaxies.extend(detect_galaxies_blob(image, **blob_kwargs))
        all_galaxies.extend(detect_galaxies_contour(image, **contour_kwargs))
        
        # Specifically look for large bright objects
        large_bright = detect_large_bright_objects(image, min_area=100)
        all_galaxies.extend(large_bright)
        
        # Remove duplicates (within 20 pixels for comprehensive - larger to avoid over-segmentation)
        # Sort by distance to process larger/more central objects first
        all_galaxies_sorted = sorted(all_galaxies, key=lambda p: (p[0]**2 + p[1]**2))
        
        unique_galaxies = []
        for x, y in all_galaxies_sorted:
            is_duplicate = False
            for ux, uy in unique_galaxies:
                # Increased distance threshold to merge over-segmented objects
                if np.sqrt((x - ux)**2 + (y - uy)**2) < 20:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_galaxies.append((x, y))
        
        return unique_galaxies
    else:
        raise ValueError(f"Unknown method: {method}. Use 'threshold', 'blob', 'contour', 'all', or 'comprehensive'")


def estimate_cluster_center(galaxy_coordinates):
    """
    Estimate cluster center from detected galaxies.
    
    Parameters
    ----------
    galaxy_coordinates : list of tuples
        List of (x, y) coordinates
        
    Returns
    -------
    tuple : (center_x, center_y)
    """
    if len(galaxy_coordinates) == 0:
        return (0, 0)
    
    coords = np.array(galaxy_coordinates)
    center_x = np.median(coords[:, 0])
    center_y = np.median(coords[:, 1])
    
    return (center_x, center_y)
