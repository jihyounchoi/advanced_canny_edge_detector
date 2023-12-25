import numpy as np

def equalize_histogram(image : np.ndarray):
    """
    Perform histogram equalization on the given image. 
    This process improves the contrast of the image and achieves a more uniform brightness distribution.

    Args:
    image: The image to apply equalization to (numpy.ndarray format).

    Returns:
    image_equalized: The image with applied histogram equalization.
    """
    
    # Calculate the histogram
    histogram, bins = np.histogram(image.flatten(), 256, [0, 256])

    # Calculate and normalize the Cumulative Distribution Function (CDF)
    cdf = histogram.cumsum()
    cdf_normalized = cdf * histogram.max() / cdf.max()

    # Mask the zeros in CDF
    cdf_masked = np.ma.masked_equal(cdf_normalized, 0)
    
    # Apply histogram equalization using the masked CDF
    cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    
    # Map the image pixels using the masked CDF
    cdf = np.ma.filled(cdf_masked, 0).astype('uint8')
    image_equalized = cdf[image]

    return image_equalized