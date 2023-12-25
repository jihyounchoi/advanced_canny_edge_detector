import numpy as np
from utils import *

def gaussian_kernel(size, sigma=1):
    """
    Create a Gaussian kernel.

    Args:
    size: Size of the kernel. It will be a square of size x size.
    sigma: Standard deviation of the Gaussian.

    Returns:
    kernel_normalized: Normalized Gaussian kernel.
    """
    
    # Define the range for each axis of the kernel.
    ax = np.linspace(-size // 2, size // 2, size)
    xx, yy = np.meshgrid(ax, ax)

    # Calculate the kernel using a 2D Gaussian function.
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

    # Normalize the kernel so that its sum is 1.
    kernel_normalized = kernel / np.sum(kernel)
    return kernel_normalized


def gaussian_blur(image, kernel_size=5, sigma=1):
    """
    Apply Gaussian blur to an image.

    Args:
    image: Image to apply blur to (numpy.ndarray format).
    kernel_size: Size of the Gaussian kernel. Default is 5.
    sigma: Standard deviation of the Gaussian kernel. Default is 1.

    Returns:
    blurred: The blurred image.
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred = convolution_2d(image, kernel)
    
    return blurred
