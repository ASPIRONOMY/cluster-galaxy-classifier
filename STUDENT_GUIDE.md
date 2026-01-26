# Student Guide: Using the Cluster Galaxy Classifier

## Quick Start for Students

This guide will help you use the **pre-trained** Cluster Galaxy Classifier to identify cluster member galaxies in your own cluster images.

## What You Need

1. **Python installed** on your computer
2. **A cluster image** (PNG or JPG format)
3. **The trained model** (already provided in the `models/` folder)

## Step 1: Install Dependencies

Open a terminal/command prompt in the project folder and run:

```bash
python -m pip install -r requirements.txt
```

## Step 2: Test on Your Cluster Image

Use the simple test script `test_your_cluster.py` (see below) or run directly:

```python
from complete_pipeline import run_complete_pipeline

# Test on your cluster image
results = run_complete_pipeline(
    image_file='path/to/your/cluster_image.png',  # Your image file
    model_dir='models',  # Uses the pre-trained model
    output_dir='results',
    detection_method='comprehensive',  # Best for detecting all objects including large ones
    confidence_threshold=0.7  # Higher = fewer false positives
)
```

## Results

After running, you'll get:
- **Visualization**: `results/your_image_classified.png`
  - Red circles = Cluster members
  - Blue circles = Non-members
- **Coordinates**: `results/your_image_members.csv`
  - X, Y coordinates of detected cluster members
  - Confidence scores

## Simple Test Script

Use `test_your_cluster.py` - just update the image path and run it!

## Tips

1. **Image format**: PNG or JPG RGB images work best
2. **Confidence threshold**:
   - `0.7` = Balanced (default, fewer false positives)
   - `0.8` = Very conservative (only high-confidence predictions)
   - `0.6` = More sensitive (may include some false positives)
3. **Detection method**: Use `'comprehensive'` for best results (detects all sizes including large objects)

## Troubleshooting

### "Model file not found"
- Make sure the `models/` folder exists with the trained model files
- Check you're in the correct directory

### "Could not load image"
- Check the image file path is correct
- Make sure the image is PNG, JPG, or FITS format

### "No galaxies detected"
- Try using `detection_method='comprehensive'`
- Lower the threshold: `detection_params={'threshold_percentile': 70}`

### "Too many false positives"
- Increase confidence threshold: `confidence_threshold=0.8`

## Questions?

Check the main `README.md` for more detailed documentation.
