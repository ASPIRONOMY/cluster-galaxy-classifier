"""
Data Preparation Script
Organizes cluster data: FITS/PNG images + CSV labels into unified dataset
"""

import numpy as np
import pandas as pd
from astropy.io import fits
import os
from pathlib import Path
import json
import cv2


def load_cluster_data(cluster_name, data_dir):
    """
    Load all data for a single cluster.
    Supports both FITS and PNG images.
    
    Parameters
    ----------
    cluster_name : str
        Name of cluster (e.g., 'cluster_001')
    data_dir : str
        Base directory containing cluster data
        
    Returns
    -------
    dict : Dictionary containing:
        - 'image': Image data (RGB array)
        - 'header': FITS header (or None for PNG)
        - 'members': DataFrame with X, Y coordinates of cluster members
        - 'non_members': DataFrame with X, Y coordinates of non-members
    """
    cluster_path = Path(data_dir) / cluster_name
    
    # Try to load image (FITS or PNG)
    rgb_image = None
    header = None
    
    # Try FITS first
    fits_files = list(cluster_path.glob('*.fits')) + list(cluster_path.glob('*.fit'))
    if fits_files:
        fits_file = fits_files[0]
        with fits.open(fits_file) as hdul:
            image_data = hdul[0].data
            header = hdul[0].header
        
        # Handle RGB FITS - could be (H, W, 3) or separate extensions
        if len(image_data.shape) == 3:
            # Already RGB
            rgb_image = image_data
        elif len(image_data.shape) == 2:
            # Single band - convert to RGB (grayscale)
            rgb_image = np.stack([image_data, image_data, image_data], axis=2)
        else:
            raise ValueError(f"Unexpected image shape: {image_data.shape}")
    
    # Try PNG/JPG if FITS not found
    if rgb_image is None:
        # Try PNG first
        png_files = list(cluster_path.glob('*.png')) + list(cluster_path.glob('*.PNG'))
        if png_files:
            png_file = png_files[0]
            # Load PNG as RGB (OpenCV loads as BGR, so convert)
            rgb_image = cv2.imread(str(png_file))
            if rgb_image is None:
                raise FileNotFoundError(f"Could not load PNG: {png_file}")
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        else:
            # Try JPG/JPEG
            jpg_files = (list(cluster_path.glob('*.jpg')) + list(cluster_path.glob('*.JPG')) +
                        list(cluster_path.glob('*.jpeg')) + list(cluster_path.glob('*.JPEG')))
            if jpg_files:
                jpg_file = jpg_files[0]
                rgb_image = cv2.imread(str(jpg_file))
                if rgb_image is None:
                    raise FileNotFoundError(f"Could not load JPG: {jpg_file}")
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            else:
                raise FileNotFoundError(f"No image file (FITS, PNG, or JPG) found in {cluster_path}")
    
    # Load members CSV
    members_file = cluster_path / f"{cluster_name}_members.csv"
    if not members_file.exists():
        # Try alternative names
        alt_names = ['members.csv', 'cluster_members.csv', f'members_{cluster_name}.csv']
        for alt in alt_names:
            alt_file = cluster_path / alt
            if alt_file.exists():
                members_file = alt_file
                break
        else:
            raise FileNotFoundError(f"Members CSV not found in {cluster_path}")
    
    members_df = pd.read_csv(members_file)
    
    # Ensure X, Y columns exist (case insensitive)
    x_col = None
    y_col = None
    for col in members_df.columns:
        col_lower = col.lower()
        if col_lower in ['x', 'xcoord', 'x_coord', 'ra', 'x_pixel']:
            x_col = col
        elif col_lower in ['y', 'ycoord', 'y_coord', 'dec', 'y_pixel']:
            y_col = col
    
    if x_col is None or y_col is None:
        raise ValueError(f"Could not find X, Y columns in {members_file}. Columns: {members_df.columns.tolist()}")
    
    members_df = members_df.rename(columns={x_col: 'X', y_col: 'Y'})
    members_df = members_df[['X', 'Y']].copy()
    members_df['cluster'] = cluster_name
    members_df['label'] = 'cluster_member'
    
    # Load non-members CSV
    non_members_file = cluster_path / f"{cluster_name}_non_members.csv"
    if not non_members_file.exists():
        alt_names = ['non_members.csv', 'foreground.csv', 'background.csv', 
                     f'non_members_{cluster_name}.csv']
        for alt in alt_names:
            alt_file = cluster_path / alt
            if alt_file.exists():
                non_members_file = alt_file
                break
        else:
            print(f"Warning: Non-members CSV not found in {cluster_path}. Creating empty DataFrame.")
            non_members_df = pd.DataFrame(columns=['X', 'Y'])
            non_members_df['cluster'] = cluster_name
            non_members_df['label'] = 'non_member'
            return {
                'image': rgb_image,
                'header': header,
                'members': members_df,
                'non_members': non_members_df,
                'cluster_name': cluster_name
            }
    
    non_members_df = pd.read_csv(non_members_file)
    
    # Find X, Y columns
    x_col = None
    y_col = None
    for col in non_members_df.columns:
        col_lower = col.lower()
        if col_lower in ['x', 'xcoord', 'x_coord', 'ra', 'x_pixel']:
            x_col = col
        elif col_lower in ['y', 'ycoord', 'y_coord', 'dec', 'y_pixel']:
            y_col = col
    
    if x_col is None or y_col is None:
        raise ValueError(f"Could not find X, Y columns in {non_members_file}")
    
    non_members_df = non_members_df.rename(columns={x_col: 'X', y_col: 'Y'})
    non_members_df = non_members_df[['X', 'Y']].copy()
    non_members_df['cluster'] = cluster_name
    non_members_df['label'] = 'non_member'
    
    return {
        'image': rgb_image,
        'header': header,
        'members': members_df,
        'non_members': non_members_df,
        'cluster_name': cluster_name
    }


def create_unified_dataset(cluster_names, data_dir, output_dir='processed_data'):
    """
    Create unified dataset from multiple clusters.
    
    Parameters
    ----------
    cluster_names : list of str
        List of cluster names to process
    data_dir : str
        Base directory containing cluster data
    output_dir : str
        Directory to save processed dataset
        
    Returns
    -------
    dict : Unified dataset information
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    all_data = {}
    all_labels = []
    
    for cluster_name in cluster_names:
        print(f"Processing {cluster_name}...")
        try:
            cluster_data = load_cluster_data(cluster_name, data_dir)
            all_data[cluster_name] = cluster_data
            
            # Combine members and non-members
            combined = pd.concat([
                cluster_data['members'],
                cluster_data['non_members']
            ], ignore_index=True)
            
            all_labels.append(combined)
            
            # Save individual cluster data
            cluster_output = output_path / cluster_name
            cluster_output.mkdir(exist_ok=True)
            
            # Save image as numpy array
            np.save(cluster_output / 'image.npy', cluster_data['image'])
            
            # Save labels
            combined.to_csv(cluster_output / 'labels.csv', index=False)
            
            print(f"  [OK] Loaded {len(cluster_data['members'])} members, "
                  f"{len(cluster_data['non_members'])} non-members")
            
        except Exception as e:
            print(f"  [ERROR] Error loading {cluster_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create unified labels file
    if all_labels:
        unified_labels = pd.concat(all_labels, ignore_index=True)
        unified_labels.to_csv(output_path / 'unified_labels.csv', index=False)
        
        # Create dataset info
        info = {
            'clusters': list(all_data.keys()),
            'total_members': int(unified_labels[unified_labels['label'] == 'cluster_member'].shape[0]),
            'total_non_members': int(unified_labels[unified_labels['label'] == 'non_member'].shape[0]),
            'total_objects': len(unified_labels)
        }
        
        with open(output_path / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n[OK] Dataset created:")
        print(f"  Total clusters: {len(all_data)}")
        print(f"  Total members: {info['total_members']}")
        print(f"  Total non-members: {info['total_non_members']}")
        print(f"  Total objects: {info['total_objects']}")
        
        return {
            'data': all_data,
            'unified_labels': unified_labels,
            'info': info,
            'output_dir': str(output_path)
        }
    else:
        raise ValueError("No data was successfully loaded!")


if __name__ == '__main__':
    # Example usage
    # Adjust these paths to match your data structure
    
    # Option 1: If your data is organized like:
    # data/
    #   cluster_001/
    #     cluster_001.png (or .fits)
    #     cluster_001_members.csv
    #     cluster_001_non_members.csv
    #   cluster_002/
    #     ...
    
    cluster_names = ['cluster_001', 'cluster_002', 'cluster_003']  # Update with your cluster names
    data_directory = 'data'  # Update with your data directory
    
    print("Creating unified dataset...")
    dataset = create_unified_dataset(cluster_names, data_directory)
    print(f"\n[OK] Dataset saved to: {dataset['output_dir']}")
