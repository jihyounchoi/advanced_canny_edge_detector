import numpy as np
from utils import *

def non_maximum_suppression_interpolation(gradient_magnitude : np.ndarray, gradient_direction : np.ndarray):
    """
    Apply non-maximum suppression with linear interpolation to thin out edges.

    Args:
    gradient_magnitude (numpy.ndarray): The gradient magnitude at each pixel.
    gradient_direction (numpy.ndarray): The gradient direction at each pixel (in radians).

    Returns:
    numpy.ndarray: The thinned-out edges after applying non-maximum suppression with interpolation.
    """
    M, N = gradient_magnitude.shape
    suppressed = np.zeros((M, N), dtype=np.float32)

    # Pad the gradient magnitude array for edge handling
    padded_magnitude = np.pad(gradient_magnitude, ((1, 1), (1, 1)), mode='constant')

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            direction = gradient_direction[i - 1, j - 1]
            # Adjust the gradient direction to be between 0 and π
            direction = direction % (np.pi)

            # Determine adjacent pixels for linear interpolation
            if 0 <= direction < np.pi / 4 or 3 * np.pi / 4 <= direction <= np.pi:
                d1 = padded_magnitude[i, j + 1]
                d2 = padded_magnitude[i, j - 1]
            elif np.pi / 4 <= direction < 3 * np.pi / 4:
                d1 = padded_magnitude[i + 1, j]
                d2 = padded_magnitude[i - 1, j]

            # Apply linear interpolation
            magnitude = padded_magnitude[i, j]
            alpha = abs(np.tan(direction))
            interpolated1 = (1 - alpha) * magnitude + alpha * d1
            interpolated2 = (1 - alpha) * magnitude + alpha * d2

            # Apply non-maximum suppression
            if magnitude >= interpolated1 and magnitude >= interpolated2:
                suppressed[i - 1, j - 1] = magnitude

    return suppressed



######################################### 실제 구현에 사용되지 않은 부분입니다 ########################################

def is_pixel_maximum_along_gradient(gradient_magnitude : np.ndarray, i, j, angle, search_range):
    """
    Determine whether the given pixel is the maximum along the gradient direction.

    Args:
    gradient_magnitude (numpy.ndarray): The gradient magnitude array.
    i, j (int): Coordinates of the pixel to be checked.
    angle (float): The gradient direction of the pixel (in radians).
    search_range (int): The range of neighboring pixels to check.

    Returns:
    bool: True if the given pixel is the maximum along the gradient direction, False otherwise.
    """

    # Check pixels within the search range
    for d in range(1, search_range + 1):
        # Calculate coordinates of neighboring pixels
        i_offset = int(d * np.sin(angle))
        j_offset = int(d * np.cos(angle))
        
        # Compare current pixel's gradient magnitude with its neighbors
        if gradient_magnitude[i, j] < gradient_magnitude[i + i_offset, j + j_offset] \
           or gradient_magnitude[i, j] < gradient_magnitude[i - i_offset, j - j_offset]:
            return False

    return True




def non_maximum_suppression_quantization(gradient_magnitude : np.ndarray, gradient_direction : np.ndarray, search_range : int = 1):
    """
    Apply non-maximum suppression (NMS) using quantized gradient directions.

    Args:
    gradient_magnitude (numpy.ndarray): The gradient magnitude array.
    gradient_direction (numpy.ndarray): The gradient direction array (in radians).
    search_range (int): The range of neighboring pixels for NMS application. Default is 1.

    Returns:
    numpy.ndarray: The array resulting from the application of NMS.
    """

    suppressed = np.zeros_like(gradient_magnitude)

    # Apply non-maximum suppression for each pixel
    for i in range(search_range, gradient_magnitude.shape[0] - search_range):
        for j in range(search_range, gradient_magnitude.shape[1] - search_range):
            current_angle = gradient_direction[i, j]
            relative_angle = quantize_angle(current_angle)

            # Check if the current pixel is the maximum along the gradient direction
            if is_pixel_maximum_along_gradient(gradient_magnitude, i, j, relative_angle, search_range):
                suppressed[i, j] = gradient_magnitude[i, j]

    return suppressed