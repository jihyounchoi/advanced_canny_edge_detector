from tqdm import tqdm
from utils import *
import numpy as np


def edge_linking_by_hysteresis(strong_edges, weak_edges, gradient_direction, mask_size=3, threshold_count=1):
    """
    Perform edge linking using hysteresis. Exclude pixels that are perpendicular to the gradient direction from being neighbors.

    Args:
    strong_edges: Binary array indicating strong edges.
    weak_edges: Binary array indicating weak edges.
    gradient_direction: Gradient direction of each pixel, represented in values ranging from -pi to pi.
    mask_size: Size of the mask, where mask_size // 2 corresponds to the search range (default: 3).
    threshold_count: Minimum number of strong edges needed for connection consideration (default: 1).

    Returns:
    edge_image: The image after applying edge linking.
    """
    
    if strong_edges.shape != weak_edges.shape:
        raise ValueError("The shape of the strong edges and the weak edges must match")
    
    if mask_size % 2 == 0:
        raise ValueError("mask_size must be an odd number")
    
    # Initialize an edge image
    edge_image = np.zeros_like(strong_edges, dtype=np.uint8)
    
    # Mark strong edges in the edge image
    edge_image[strong_edges] = 255
    
    # Define the range for the mask
    mask_range = mask_size // 2
    
    # Iterate through the image
    for i in range(mask_range, edge_image.shape[0] - mask_range):
        for j in range(mask_range, edge_image.shape[1] - mask_range):
            if weak_edges[i, j]:
                # Extract the neighborhood of the current pixel
                neighborhood = edge_image[i - mask_range : i + mask_range + 1, j - mask_range : j + mask_range + 1]
                
                # edge_direction을 중심으로, [i, j] 기준으로 gradient 방향에 해당하는 (즉, edge에 수직한) 픽셀들은 제거합니다 (gradient quantization을 사용합니다).
                if quantize_angle(gradient_direction[i, j]) == 0:  # Horizontal
                    neighborhood[mask_range, :] = 0
                elif quantize_angle(gradient_direction[i, j]) == np.pi / 2:  # Vertical
                    neighborhood[:, mask_range] = 0
                elif quantize_angle(gradient_direction[i, j]) == np.pi / 4:  # 45-degree
                    for k in range(-mask_range, mask_range + 1):
                        if 0 <= i + k < edge_image.shape[0] and 0 <= j + k < edge_image.shape[1]:
                            neighborhood[i + k - (i - mask_range), j + k - (j - mask_range)] = 0
                elif quantize_angle(gradient_direction[i, j]) == 3 * np.pi / 4:  # 135-degree
                    for k in range(-mask_range, mask_range + 1):
                        if 0 <= i + k < edge_image.shape[0] and 0 <= j - k < edge_image.shape[1]:
                            neighborhood[i + k - (i - mask_range), j - k - (j - mask_range)] = 0
                
                # Count the number of strong edges in the neighborhood
                strong_edge_count = np.sum(neighborhood == 255)
                
                # If the count is above the threshold, mark the current weak edge as strong
                if strong_edge_count >= threshold_count:
                    edge_image[i, j] = 255

    return edge_image


######################################### 실제 구현에서 사용되지 않은 코드들입니다 #########################################



'''
calculate_weight_distance, calculate_weight_coverage, calculate_weight_dot_product 함수는 super_edge_linking_by_hysteresis 함수에서 사용됩니다.
'''

def calculate_weight_distance(pixel_position_1, pixel_position_2, mask_size):
    """
    Calculate the distance between two pixels and normalize it based on the mask_size.

    Args:
    pixel_position_1: (x, y) coordinates of the first pixel.
    pixel_position_2: (x, y) coordinates of the second pixel.
    mask_size: Size of the mask (assumed to be square).

    Returns:
    normalized_distance: Distance normalized between 0 and 1.
    """

    # 두 픽셀 위치 사이의 유클리드 거리 계산
    distance = np.sqrt((pixel_position_1[0] - pixel_position_2[0])**2 + (pixel_position_1[1] - pixel_position_2[1])**2)
    
    # 거리를 정규화 (최대 거리는 마스크의 대각선 길이로 가정)
    normalized_distance = distance / ((mask_size // 2) * np.sqrt(2))
    return normalized_distance




def calculate_weight_coverage(line_crossing_point, line_gradient, pixel_position):
    """
    Calculate the distance between a specific line and a pixel position. Returns True if the distance is within sqrt(2), else False.

    Args:
    line_crossing_point: (x, y) point that the line crosses.
    line_gradient: Gradient of the line (-pi to pi, perpendicular direction).
    pixel_position: (x, y) position of the pixel to calculate the distance from.

    Returns:
    True or False: True if the pixel is within sqrt(2) distance from the line, else False.
    """

    x0, y0 = line_crossing_point
    px, py = pixel_position

    # 직선이 수직인 경우 특별 처리
    if line_gradient == np.pi / 2 or line_gradient == -np.pi / 2:
        distance = abs(px - x0)
    else:
        # 직선의 기울기와 y절편 계산
        m = -1 / np.tan(line_gradient)
        c = y0 - m * x0

        # 픽셀과 직선 사이의 거리 계산
        distance = abs((m * px - py + c) / np.sqrt(m**2 + 1))

    # 거리가 sqrt(2) 이내인지 여부 반환
    return distance < np.sqrt(2)



def calculate_weight_dot_product(gradient_direction_of_vector_to_projected, gradient_direction_of_vector_to_project_from, gradient_magintude_of_vector_to_project_from):
    """
    Calculate the dot product of two vectors and return its absolute value.

    Args:
    gradient_direction_of_vector_to_projected: Direction of the first vector (-pi to pi).
    gradient_direction_of_vector_to_project_from: Direction of the second vector (-pi to pi).
    gradient_magintude_of_vector_to_project_from: Magnitude of the second vector.

    Returns:
    The absolute value of the dot product.
    """

    # 첫 번째 벡터 생성 (단위 벡터)
    vector_to_projected = np.array([np.cos(gradient_direction_of_vector_to_projected), np.sin(gradient_direction_of_vector_to_projected)])

    # 두 번째 벡터 생성 (주어진 크기)
    vector_to_project_from = gradient_magintude_of_vector_to_project_from * np.array([np.cos(gradient_direction_of_vector_to_project_from), np.sin(gradient_direction_of_vector_to_project_from)])

    # 내적(dot product) 계산
    dot_product = np.dot(vector_to_projected, vector_to_project_from)
    
    # 내적의 절대값 반환
    return abs(dot_product)




def super_edge_linking_by_hysteresis(strong_edges, weak_edges, gradient_magnitude, gradient_direction, threshold_percentage, mask_size, weight_coverage = 1, weight_distance = 1, weight_dotproduct = 1):
    
    """
    Perform advanced edge linking using hysteresis. This method considers the direction and magnitude of gradients.

    Args:
    strong_edges: Binary array indicating strong edges.
    weak_edges: Binary array indicating weak edges.
    gradient_magnitude: Gradient magnitude at each pixel.
    gradient_direction: Gradient direction at each pixel.
    threshold_percentage: Percentage threshold to consider when adding pixels to final edges.
    mask_size: Size of the mask.
    weight_coverage, weight_distance, weight_dotproduct: Weights for coverage, distance, and dot product.

    Returns:
    final_edges: The final edges after applying advanced edge linking.

    0. final_edges 배열을 생성합니다 (초기값은 strong_edges와 동일합니다).
    1. weak_edges에 포함된 각 픽셀에 대해, 아래의 과정을 수행합니다.
        1-1. weak_edges[i, j]가 True인지 확인합니다.
        1-2. [i, j] 지점을 중심으로, 한 변의 길이가 mask_size인 정사각형 마스크를 생성합니다.
        1-3. [i, j]기준으로 gradient_direction 방향 (방향 및 역방향 모두)에 걸쳐 있고 마스크 내에 있는 픽셀들을 기준으로, 다음을 계산합니다 (angle quantization을 사용하지 않습니다).
            1-3-1. 해당 픽셀이 gradient_direction을 향하는 직선과 얼마나 걸쳐 있는지 계산합니다. weak_edges[i, j]에서 gradient에 수직인 방향으로 직선을 그었을 때, 판단하고자 하는 픽셀의 "중심점"과 직선과의 거리를 기준으로 weight를 결정합니다. 이때 거리가 0 ~ sqrt(2) 사이인 경우 지난다고 할 수 있으며, 해당 값에 1 / sqrt(2)를 곱해서 0 ~ 1의 weight로 사용합니다.  -> 이 값을 weight_coverage라고 부릅니다.
            1-3-2. 해당 픽셀이 [i, j] 와 얼마나 멀리 있는지 확인하고, 거리의 역수를 계산한 후 0 ~ 1사이의 값을 갖도록 normalization합니다. -> 이 값을 weight_distance라고 부릅니다.
            1-3-3. 해당 픽셀의 gradient의 [i, j]의 gradient 방향에 대한 내적을 계산합니다 (normalization을 사용하지 않습니다.) -> 이 값을 weight_projection이라고 부릅니다.
            1-3-4. weight_coverage, weight_distance, weight_projection을 곱하여 해당 픽셀의 weight를 계산합니다.
        1-4. 모든 픽셀의 weight를 더하여 최종 score를 배열에 저장합니다.
    2. 계산된 최종 score들 중, 상위 threshold_percentage에 해당하는 픽셀들을 final_edges에 추가합니다.
    3. 결과를 리턴합니다.
    """
    
    # Padding을 적용하여 가장자리 부분도 고려
    pad_width = mask_size // 2
    padded_strong_edges = np.pad(strong_edges, pad_width, mode='constant', constant_values=0)
    padded_weak_edges = np.pad(weak_edges, pad_width, mode='constant', constant_values=0)
    padded_gradient_magnitude = np.pad(gradient_magnitude, pad_width, mode='constant', constant_values=0)
    padded_gradient_direction = np.pad(gradient_direction, pad_width, mode='constant', constant_values=0)

    final_edges = np.copy(padded_strong_edges)
    scores = np.zeros_like(padded_weak_edges, dtype=float)

    for i in tqdm(range(0, padded_weak_edges.shape[0] - 2 * pad_width), desc="Super edge linking", total=padded_weak_edges.shape[0] - 2 * pad_width):
        for j in range(0, padded_weak_edges.shape[1] - 2 * pad_width):
            
            if padded_weak_edges[i + pad_width, j + pad_width]:
                
                total_score = 0
                
                for di in range(-pad_width, pad_width + 1):
                    for dj in range(-pad_width, pad_width + 1):
                        
                        ni, nj = i + di + pad_width, j + dj + pad_width
                        
                        if padded_weak_edges[ni, nj] == 0: 
                            continue
                        
                        distance = calculate_weight_coverage(line_crossing_point=(i, j), line_gradient=padded_gradient_direction[i + pad_width, j + pad_width], pixel_position=(ni, nj))
                        if not distance:
                            continue
                        
                        coverage = calculate_weight_distance((ni, nj), (i + pad_width, j + pad_width), mask_size)
                        dotproduct = calculate_weight_dot_product(padded_gradient_direction[i + pad_width, j + pad_width], padded_gradient_direction[ni, nj], padded_gradient_magnitude[ni, nj])
                        
                        total_score += weight_coverage * coverage + weight_distance * distance + weight_dotproduct * dotproduct
                
                scores[i + pad_width, j + pad_width] = total_score
    

    valid_scores = scores[padded_weak_edges]
    threshold_value = np.percentile(valid_scores, 100 - threshold_percentage)
    
    print(f"threshold_value: {threshold_value}")
    
    # Add pixels to final_edges where score is above threshold
    final_edges[padded_weak_edges & (scores >= threshold_value)] = 1
    
    print(f"score matrix : {scores[pad_width:-pad_width, pad_width:-pad_width]}")

    # Remove padding
    final_edges = final_edges[pad_width:-pad_width, pad_width:-pad_width]
    
    return final_edges




def recursive_edge_linking(strong_edges, weak_edges, search_range=1):
    """
    Perform recursive edge linking. It starts from strong edges and recursively adds connected weak edges.

    Args:
    strong_edges: Binary array indicating strong edges.
    weak_edges: Binary array indicating weak edges.
    search_range: Range to search for connecting weak edges.

    Returns:
    edge_image: Image with edges after applying recursive edge linking.
    """
    
    # Initialize an edge image with strong edges
    edge_image = np.copy(strong_edges)
    
    # Initialize a list of points to be checked
    points_to_check = set(zip(*np.where(strong_edges)))

    while points_to_check:
        i, j = points_to_check.pop()

        # Check the neighborhood within the search range
        for di in range(-search_range, search_range + 1):
            for dj in range(-search_range, search_range + 1):
                ni, nj = i + di, j + dj
                if (0 <= ni < weak_edges.shape[0]) and (0 <= nj < weak_edges.shape[1]):
                    # If a weak edge is found that is not already a strong edge
                    if weak_edges[ni, nj] and not edge_image[ni, nj]:
                        # Add it as a new strong edge and add the point to the check list
                        edge_image[ni, nj] = 1
                        points_to_check.add((ni, nj))

    return edge_image.astype(np.uint8) * 255
