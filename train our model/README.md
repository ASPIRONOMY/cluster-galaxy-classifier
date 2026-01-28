# Coordinate Widget - Train Our Model

This folder contains an interactive image annotation tool for selecting galaxy cluster members and background objects.

## Overview

The main script is `image_click_coordinates.py`. It opens a PNG image and lets you:

- Click on **cluster members** (galaxy members) and record their coordinates.
- Click on **background objects** (cluster non-members) and record their coordinates.
- Zoom in and out to mark objects precisely.
- Save separate CSV files and marked images for each class.

## Features

- **Two modes**:
  - Galaxy members (cluster members) – **red markers**
  - Background objects (cluster non-members) – **blue markers**
- **Keyboard shortcuts**:
  - `M` – switch to Galaxy Members mode
  - `B` – switch to Background Objects mode
- **Zoom & pan**:
  - Mouse wheel to zoom in/out
  - Scrollbars to move around when zoomed
- **Outputs** (per run):
  - A new folder `cluster_XXX` (e.g., `cluster_001`, `cluster_002`, ...)
  - `image.png` – copy of the original image
  - `cluster_members.csv` – X,Y coordinates of members
  - `cluster_non_members.csv` – X,Y coordinates of non-members
  - `image_marked_members.png` – image with red markers (if any members)
  - `image_marked_non_members.png` – image with blue markers (if any non-members)

## Requirements

- Python 3.x
- `tkinter` (usually included with standard Python on Windows)
- `Pillow` (PIL fork)

Install dependencies (from a terminal):

```bash
pip install Pillow
```

## Usage

From a terminal (PowerShell on Windows):

```bash
cd C:\Users\Naora\coordnate_widget_new_new
python image_click_coordinates.py
```

You can also pass an image path directly:

```bash
python image_click_coordinates.py path\to\your_image.png
```

### Annotating procedure

1. Start in **Galaxy Members** mode (red markers).
2. Click on all cluster member galaxies.
3. Press **B** or click the *Background Objects (B)* button.
4. Click on all background / non-member objects.
5. Press *Finish Current Session* (optional) or simply close the window.
6. Check the newly created `cluster_XXX` folder for CSVs and marked images.

These CSVs and images can then be used to train and validate your galaxy cluster classifier.

---

## Google Colab (for students)

If you don’t have Python locally or prefer to work in the browser, use the **Colab notebook**:

- **File:** `cluster_annotation_colab.ipynb`
- **How to use:**
  1. Upload `cluster_annotation_colab.ipynb` to [Google Colab](https://colab.research.google.com/) (File → Upload notebook), or open it from Google Drive.
  2. Run the cells in order: upload your image, then click on the image to mark members (red) and background objects (blue).
  3. Use the buttons below the plot to switch between **Galaxy Members** and **Background Objects**.
  4. When finished, run the last cell to save and download a zip of your `cluster_XXX` folder (CSVs and marked images).

The Colab version uses the same logic as the desktop script but runs entirely in the browser (no tkinter).

