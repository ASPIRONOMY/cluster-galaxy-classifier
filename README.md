# Cluster Galaxy Classifier

A computer vision system to automatically identify cluster member galaxies in astronomical images.
To use the trained model on your own cluster image go to : https://colab.research.google.com/github/ASPIRONOMY/cluster-galaxy-classifier/blob/main/student_notebook_colab.ipynb
 

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ASPIRONOMY/cluster-galaxy-classifier/blob/main/train_our_model/cluster_annotation_colab.ipynb)

## Overview

This system:
1. **Detects** all galaxies in an image (finds many more than your labeled ~300)
2. **Extracts** features (color, shape, brightness, position)
3. **Classifies** each detected galaxy as cluster member or non-member
4. **Outputs** marked image + CSV with (X, Y) coordinates of cluster members

### Key Features

✅ **Works on any image file size** - automatically adapts to different pixel scales  
✅ **Supports PNG and FITS formats** - use RGB PNG images or FITS files  
✅ **Generalizes to new clusters** - train on 3 clusters, test on unseen ones  
✅ **Handles different pixel sizes** - adaptive feature radius  
✅ **Detects all galaxies** - not just your labeled ~300, then classifies them

## Project Structure

```
cluster_classifier/
├── data_preparation.py      # Organize FITS/PNG images + CSV labels
├── feature_extraction.py    # Extract CV features from galaxies
├── galaxy_detection.py      # Detect galaxies in images
├── train_classifier.py      # Train classification model
├── complete_pipeline.py     # End-to-end pipeline
├── workflow.py              # Main workflow script
└── README.md                # This file
```

## Data Requirements

### Training Data Structure

Your data should be organized like this:

```
data/
├── cluster_001/
│   ├── cluster_001.png          # RGB PNG image (or .fits file)
│   ├── cluster_001_members.csv   # X, Y coordinates of cluster members
│   └── cluster_001_non_members.csv  # X, Y coordinates of non-members
├── cluster_002/
│   ├── cluster_002.png
│   ├── cluster_002_members.csv
│   └── cluster_002_non_members.csv
└── cluster_003/
    ...
```

### CSV Format

**members.csv** and **non_members.csv** should have X, Y columns:
```csv
X,Y
123.5,456.7
234.1,567.8
...
```

Column names can be: `X`, `x`, `X_coord`, `x_pixel`, `RA` (for X)  
and: `Y`, `y`, `Y_coord`, `y_pixel`, `DEC` (for Y)

## Quick Start

### 1. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Prepare Your Data

Make sure your data is organized as described above. You can use either:
- **PNG RGB images** (recommended for easier workflow)
- **FITS files** (RGB or single-band)

### 3. Run the Workflow

Edit `workflow.py` and update:
- `CLUSTER_NAMES`: List of your cluster folder names
- `DATA_DIR`: Path to your data directory

Then run:
```bash
python workflow.py
```

This will:
1. Load and organize your data
2. Extract features for all labeled galaxies
3. Train a classifier
4. Save the model

### 4. Test on New Images (Including Unseen Clusters!)

```python
from complete_pipeline import run_complete_pipeline

# Test on a completely new cluster the model has never seen!
results = run_complete_pipeline(
    image_file='path/to/new_cluster.png',  # or .fits - Can be any size, any pixel scale
    model_dir='models',
    model_type='random_forest',
    output_dir='results',
    feature_radius=0.01,  # Adapts to image size (1% of image)
    adaptive_radius=True,  # Handles different pixel scales
    confidence_threshold=0.5
)
```

**Note**: The system will detect ALL galaxies in the image (could be 500-2000+), then classify which are cluster members. Your ~300 labeled galaxies are used for training, not as a limit on detection.

This will:
- Detect galaxies
- Classify them
- Save marked image (`*_classified.png`)
- Save coordinates (`*_members.csv`)

## Step-by-Step Usage

### Step 1: Data Preparation

```python
from data_preparation import create_unified_dataset

cluster_names = ['cluster_001', 'cluster_002', 'cluster_003']
dataset = create_unified_dataset(cluster_names, data_dir='data', output_dir='processed_data')
```

### Step 2: Feature Extraction

```python
from feature_extraction import extract_features_for_all_galaxies, add_position_features
import pandas as pd
import numpy as np

# Load image and labels
image = np.load('processed_data/cluster_001/image.npy')
labels = pd.read_csv('processed_data/cluster_001/labels.csv')

# Extract features
features_df = extract_features_for_all_galaxies(image, labels, radius=10)

# Add position features
center_x, center_y = labels['X'].median(), labels['Y'].median()
features_df = add_position_features(features_df, center_x, center_y)
```

### Step 3: Train Model

```python
from train_classifier import train_classifier

results = train_classifier(
    features_file='features.csv',
    output_dir='models',
    model_type='random_forest'  # or 'svm'
)
```

### Step 4: Use on New Images

```python
from complete_pipeline import run_complete_pipeline

results = run_complete_pipeline(
    image_file='new_cluster.png',  # or .fits
    model_dir='models',
    detection_method='threshold',  # or 'blob', 'contour', 'all'
    confidence_threshold=0.5
)
```

## Features Extracted

The system extracts **30+ features** for each galaxy:

### Color Features (Key for red cluster galaxies)
- Mean R, G, B values
- Color ratios (R/G, R/B, G/B)
- Redness index
- Color differences

### Brightness Features
- Mean, max, std, median brightness
- Center vs edge brightness
- Brightness gradients

### Shape Features
- Area, perimeter, circularity
- Eccentricity, solidity, extent
- Major/minor axis, aspect ratio

### Texture Features
- Contrast, dissimilarity, homogeneity
- Energy, correlation (from GLCM)

### Position Features
- Distance from cluster center
- Normalized distance

## Model Types

### Random Forest (Recommended)
- Fast training
- Handles non-linear relationships
- Provides feature importance
- Good for ~300 samples

### SVM
- Can be more accurate with small datasets
- Requires feature scaling
- Slower for large datasets

## Detection Methods

### Threshold
- Simple and fast
- Good for extended objects
- Parameters: `threshold_percentile`, `min_area`, `max_area`

### Blob Detection
- Good for circular/elliptical objects
- Parameters: `min_sigma`, `max_sigma`, `threshold`

### Contour
- Good for irregular shapes
- Parameters: `threshold_percentile`, `min_area`

### All (Combined)
- Uses all methods and removes duplicates
- Most comprehensive but slowest

## Output Files

### Training Output
- `models/random_forest_model.pkl` - Trained model
- `models/random_forest_scaler.pkl` - Feature scaler
- `models/random_forest_results.json` - Performance metrics
- `models/feature_names.json` - Feature list

### Testing Output
- `results/*_classified.png` - Image with marked galaxies
  - Red circles: Cluster members
  - Blue circles: Non-members
- `results/*_members.csv` - Coordinates of cluster members
  - Columns: X, Y, confidence

## Tips

1. **Feature Radius**: 
   - Use `feature_radius=0.01` (1% of image size) for adaptive scaling
   - Works across different pixel sizes automatically
   - Adjust to 0.02-0.03 for larger galaxies, 0.005 for smaller

2. **Different Pixel Sizes**: 
   - Always use `adaptive_radius=True` (default)
   - System automatically adapts to each image's scale

3. **Confidence Threshold**: 
   - `0.5` = balanced (default)
   - `0.7` = more conservative (fewer false positives)
   - `0.3` = more sensitive (more detections)

4. **Detection Method**: Try different methods if detection is poor
5. **Model Type**: Start with Random Forest, try SVM if needed

6. **Testing on New Clusters**: 
   - The model is designed to generalize!
   - Train on 3 clusters, test on completely new ones
   - Model learns generalizable features (redness, shape, etc.)

7. **Image Format**:
   - PNG RGB images work great and are easier to work with
   - FITS files are also supported (RGB or single-band)
   - The system automatically detects the format

## Troubleshooting

### "No galaxies detected"
- Try different detection method
- Adjust `threshold_percentile` (lower = more sensitive)
- Check if image is properly normalized

### "Low accuracy"
- Check if you have enough training data
- Verify labels are correct
- Try different feature radius
- Check feature importance to see what matters

### "Model file not found"
- Make sure you've run training first
- Check `model_dir` path is correct

### "Could not load image"
- Make sure image file exists
- Check file format (PNG or FITS)
- Verify file is not corrupted

## Next Steps

After training:
1. Test on your 3 clusters to validate
2. Adjust parameters if needed
3. Collect more data with students
4. Retrain with larger dataset
5. Create teaching notebooks for students
