import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import copy


def thresholding(image_suppressed: np.ndarray, low_threshold: float, high_threshold: float) -> tuple:
    """
    Apply basic thresholding to identify strong and weak edges in an image.

    Args:
    image_suppressed (numpy.ndarray): The image after applying non-maximum suppression.
    low_threshold (float): The lower threshold for weak edges.
    high_threshold (float): The higher threshold for strong edges.

    Returns:
    tuple: A tuple containing two numpy arrays for strong and weak edges.
    """
    strong_edges = (image_suppressed >= high_threshold)
    weak_edges = (image_suppressed >= low_threshold) & (image_suppressed < high_threshold)

    return strong_edges, weak_edges


def quantize_angle(angle: float) -> float:
    """
    Quantize an angle to one of the four major directions (0, 45, 90, 135 degrees).

    Args:
    angle (float): The angle to quantize (in radians).

    Returns:
    float: The quantized angle.
    """
    absolute_angle = np.abs(angle)
    
    # Quantize the angle to one of four major directions
    # Horizontal (0 degrees)
    if absolute_angle < (np.pi / 4) * 0.5 or absolute_angle >= (np.pi - (np.pi / 4) * 0.5):
        return 0  
    
    # 45-degree edge
    elif (np.pi / 4) * 0.5 <= absolute_angle < (np.pi / 2 - (np.pi / 4) * 0.5):
        return np.pi / 4  
    
    # Vertical (90 degrees)
    elif (np.pi / 2 - (np.pi / 4) * 0.5) <= absolute_angle < (np.pi / 2 + (np.pi / 4) * 0.5):
        return (np.pi / 4) * 2  
    
    # 135-degree edge
    elif (np.pi / 2 + (np.pi / 4) * 0.5) <= absolute_angle < (np.pi - (np.pi / 4) * 0.5):
        return (np.pi / 4) * 3  
    
    else:
        raise ValueError(f"Angle {absolute_angle} is not supported.")


def convolution_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply 2D convolution to an image using a given kernel.

    Args:
    image (numpy.ndarray): The input image.
    kernel (numpy.ndarray): The convolution kernel.

    Returns:
    numpy.ndarray: The convolved image.
    """

    if len(image.shape) == 2:
        image = image[..., np.newaxis]
        m_i, n_i, c_i = image.shape
        
    else:
        raise Exception('Shape of image not supported')

    m_k, n_k = kernel.shape

    y_strides = m_i - m_k + 1  # possible number of strides in y direction
    x_strides = n_i - n_k + 1  # possible number of strides in x direction

    img = image.copy()
    output_shape = (m_i-m_k+1, n_i-n_k+1, c_i)
    output = np.zeros(output_shape, dtype=np.float32)

    count = 0  # taking count of the convolution operation being happening

    output_tmp = output.reshape(
        (output_shape[0]*output_shape[1], output_shape[2])
    )

    for i in range(y_strides):
        for j in range(x_strides):
            for c in range(c_i):
                sub_matrix = img[i:i+m_k, j:j+n_k, c]

                output_tmp[count, c] = np.sum(sub_matrix * kernel)

            count += 1

    output = output_tmp.reshape(output_shape)
    
    # 2D convolution이므로, channel 정보를 제거
    output = output[..., 0]

    return output


def resize(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Resize an image while maintaining its aspect ratio.

    Args:
    image (numpy.ndarray): The input image.
    target_width (int): The desired width in pixels.
    target_height (int): The desired height in pixels.

    Returns:
    numpy.ndarray: The resized image.
    """
    
    # Check if the image is a Numpy Array and convert it to a PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    image_resized = image.resize((target_width, target_height))

    return np.array(image_resized)


def show(*arrays: np.ndarray):
    """
    Show multiple images represented as numpy arrays in a single plot.

    Args:
    arrays (numpy.ndarray): A variable number of numpy arrays representing images.
    """
    num_images = len(arrays)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    if num_images == 1:
        axes = [axes]

    for ax, array in zip(axes, arrays):
        ax.imshow(array, cmap='gray')
        ax.axis('off')

    plt.show()



def calculate_gradient(image: np.ndarray) -> tuple:
    """
    Calculate the gradient magnitude and direction of an image.

    Args:
    image (numpy.ndarray): The input image.

    Returns:
    tuple: A tuple containing the gradient magnitude and direction arrays.
    """
    
    sobel_x = [[-1, 0, 1], 
               [-2, 0, 2], 
               [-1, 0, 1]]
    
    sobel_y = [[-1, -2, -1], 
               [0, 0, 0], 
               [1, 2, 1]]
    
    sobel_x = np.array(sobel_x, dtype=np.float32)
    sobel_y = np.array(sobel_y, dtype=np.float32)
    
    gradient_x = convolution_2d(image, sobel_x)
    gradient_y = convolution_2d(image, sobel_y)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    return gradient_magnitude, gradient_direction


def ignore_isolated_value(image_edges: np.ndarray, search_range: int = 1) -> np.ndarray:
    """
    Ignore isolated values in an edge image based on a search range.

    Args:
    image_edges (numpy.ndarray): The edge image.
    search_range (int): The search range for neighboring pixels.

    Returns:
    numpy.ndarray: The modified edge image.
    """
    
    min_neighbors = search_range * 2
    
    fixed_image_edges = copy.deepcopy(image_edges)  # 입력 배열 복사
    
    for i in range(search_range, fixed_image_edges.shape[0] - search_range):
        for j in range(search_range, fixed_image_edges.shape[1] - search_range):
            if image_edges[i, j] != 0:  # 변경된 조건: 에지 픽셀이 0보다 큰 경우
                if np.sum(image_edges[i-search_range:i+search_range+1, j-search_range:j+search_range+1]) < min_neighbors * 255:
                    fixed_image_edges[i, j] = 0

    return fixed_image_edges



######################################## 실제 구현에 사용되지 않은 부분입니다 #################################

# Block-by-block 처리를 위해 구현되었으나, 사용되지 않았습니다.
def segment_image(image: np.ndarray, segment_size: int) -> list:
    """
    Segment the image into smaller regions.

    Args:
    image (numpy.ndarray): The input image.
    segment_size (int): The size of each segment.

    Returns:
    list: A list of tuples containing the segment's coordinates and the segment itself.
    """
    segments = []
    for i in range(0, image.shape[0], segment_size):
        for j in range(0, image.shape[1], segment_size):
            segment = image[i:i+segment_size, j:j+segment_size]
            segments.append((i, j, segment))
    return segments