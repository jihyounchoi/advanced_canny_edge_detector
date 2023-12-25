import numpy as np
from utils import *
from tqdm import tqdm


def otsu_two_thresholds(image_suppressed: np.ndarray) -> tuple:
    """
    Compute two thresholds using Otsu's method for an image with suppressed gradients.

    Args:
    image_suppressed (numpy.ndarray): The image after applying non-maximum suppression.

    Returns:
    tuple: A tuple containing the lower and upper thresholds.
    """
    pixels = image_suppressed.flatten()
    histogram, bin_edges = np.histogram(pixels, bins=256, range=(0, image_suppressed.max()))
    histogram = histogram.astype(float) / histogram.sum()

    best_between_class_variance = 0
    best_threshold1 = 0
    best_threshold2 = 0

    total_mean = np.sum(histogram * np.arange(256))
    total_weight = histogram.sum()

    for t1 in range(256):
        weight1 = np.sum(histogram[:t1])
        mean1 = np.sum(histogram[:t1] * np.arange(t1)) / weight1 if weight1 > 0 else 0

        for t2 in range(t1 + 1, 256):
            weight2 = np.sum(histogram[t1:t2])
            mean2 = np.sum(histogram[t1:t2] * np.arange(t1, t2)) / weight2 if weight2 > 0 else 0

            weight3 = total_weight - weight1 - weight2
            mean3 = (total_mean - mean1 * weight1 - mean2 * weight2) / weight3 if weight3 > 0 else 0

            between_class_variance = (
                weight1 * (mean1 - total_mean) ** 2 +
                weight2 * (mean2 - total_mean) ** 2 +
                weight3 * (mean3 - total_mean) ** 2
            )

            if between_class_variance > best_between_class_variance:
                best_between_class_variance = between_class_variance
                best_threshold1 = t1
                best_threshold2 = t2
                
    low_high = np.array([best_threshold1, best_threshold2])

    return low_high.min(), low_high.max()


############################################# 실제 구현에 사용되지 않은 부분입니다 #########################################

def otsu_threshold(image: np.ndarray) -> float:
    """
    Compute Otsu's threshold for an image.

    Args:
    image (numpy.ndarray): The input image for threshold calculation.

    Returns:
    float: The computed threshold normalized between 0 and 1.
    """
    # Flatten the image and calculate the histogram
    pixels = image.flatten()
    histogram, bin_edges = np.histogram(pixels, bins=256, range=(0, 256))

    # Normalize the histogram
    histogram = histogram.astype(float) / histogram.sum()

    # Cumulative sum and cumulative mean of the histogram
    cumulative_sum = np.cumsum(histogram)
    cumulative_mean = np.cumsum(histogram * np.arange(256))

    # Full-image mean
    full_image_mean = cumulative_mean[-1]

    # Between-class variance for all possible thresholds
    between_class_variance = ((full_image_mean * cumulative_sum - cumulative_mean) ** 2) / (cumulative_sum * (1 - cumulative_sum))

    # Handle NaN and inf values
    between_class_variance = np.nan_to_num(between_class_variance)

    # Find the threshold that maximizes the between-class variance
    optimal_threshold = np.argmax(between_class_variance)

    return optimal_threshold / 255.0




def percentile_based_two_thresholds(gradient_magnitude: np.ndarray, low_percentile: float, high_percentile: float) -> tuple:
    """
    Determine two thresholds based on percentiles of the gradient magnitudes.

    Args:
    gradient_magnitude (numpy.ndarray): The gradient magnitude of the image.
    low_percentile (float): The lower percentile for thresholding.
    high_percentile (float): The upper percentile for thresholding.

    Returns:
    tuple: A tuple containing the lower and upper thresholds.
    """
    
    # Determine thresholds
    low_threshold = np.percentile(gradient_magnitude, low_percentile)
    high_threshold = np.percentile(gradient_magnitude, high_percentile)

    return low_threshold, high_threshold


def pixel_adaptive_double_thresholding(image_suppressed: np.ndarray, mask_size: int = 3) -> tuple:
    """
    Perform pixel-adaptive double thresholding using local Otsu's thresholds.

    Args:
    image_suppressed (numpy.ndarray): The image after applying non-maximum suppression.
    mask_size (int): Size of the local mask to apply Otsu's method.

    Returns:
    tuple: A tuple containing numpy arrays for strong and weak edges.
    """
    padded_image = np.pad(image_suppressed, pad_width=mask_size//2, mode='constant')
    
    strong_edges = np.zeros_like(image_suppressed, dtype=bool)
    weak_edges = np.zeros_like(image_suppressed, dtype=bool)

    for i in tqdm(range(image_suppressed.shape[0]), desc="Pixel adaptive hysteresis thresholding", total = image_suppressed.shape[0]):
        for j in range(image_suppressed.shape[1]):
            
            # Apply Otsu's method to the local mask
            mask = padded_image[i:i+mask_size, j:j+mask_size]
            
            # Calculate the thresholds
            low_threshold, high_threshold = otsu_two_thresholds(mask)
            
            # Calculate the threshold values
            high_thresh_val = np.max(mask) * high_threshold
            low_thresh_val = np.max(mask) * low_threshold
            
            # Determine strong and weak edges
            if image_suppressed[i, j] >= high_thresh_val:
                strong_edges[i, j] = True
            elif image_suppressed[i, j] >= low_thresh_val:
                weak_edges[i, j] = True

    return strong_edges, weak_edges