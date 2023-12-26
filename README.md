# Advanced Canny Edge Detector

## Overview
This project aims to enhance the cv2.Canny() edge detector by addressing its limitations and implementing a more powerful edge detection algorithm. The enhancements include an algorithm that works well without the need for frequent parameter adjustments and produces thinner edges while maintaining the edge structure.

## Key Features
1. Pre-application of histogram equalization before Gaussian blur to improve image contrast for easier edge detection.
2. Automated double thresholding using Otsu's method.
3. Adjustable search range in edge linking and a unique implementation approach.
4. Introduction of `ignore_isolated_value()` method for post-processing to correct mistakenly selected edges.

## File Structure
- `/dataset`: Contains image files used for edge detection.
- `edge_linker.py`, `gaussian_blur.py`, `histogram_equalization.py`, `non_maximum_suppression.py`, `thresholding.py`: Utility modules for implementing various aspects of the Canny edge detector.
- `utils.py`: Contains globally used utilities.

## Requirements
- PIL
- Numpy
- Matplotlib (for visualization)
- cv2.Canny (used only for comparison, not included in the implementation)

## Usage
To use the advanced edge detector, follow these steps:
1. Import the necessary modules and utilities.
2. Load an image and convert it to grayscale.
3. Apply the `advanced_edge_detector()` function to the image.
4. Optionally, visualize the results using Matplotlib.

Example Code:
```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import utils
from gaussian_blur.py import gaussian_blur
from histogram_equalization.py import equalize_histogram
from non_maximum_suppression.py import non_maximum_suppression_interpolation
from thresholding.py import otsu_two_thresholds, thresholding
from edge_linker.py import edge_linking_by_hysteresis
```

## Load and process the image
image_path = "dataset/your_image.png"
image_gray = Image.open(image_path).convert("L")
image_processed = advanced_edge_detector(np.array(image_gray))

## Visualize the result
plt.imshow(image_processed, cmap='gray')
plt.show()

## Note
The edge_linker, non_maximum_suppression, and thresholding modules contain various methods explored during the implementation, providing insight into the development process.  
Feel free to use and modify this template as needed for your project.
