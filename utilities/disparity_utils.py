import numpy as np
import matplotlib.pyplot as plt
import re
import cv2
import json
import open3d as o3d
import matplotlib.pyplot as plt

def load_pfm(file):
    """
    Load a PFM file.
    
    Args:
    - file: path to the PFM file.

    Returns:
    - data: loaded data from PFM.
    - scale: scale factor.
    """

    with open(file, 'rb') as f:
        header = f.readline().rstrip()
        color = False
        if header == b'PF':
            color = True
        elif header != b'Pf':
            raise Exception('Not a PFM file.')

        # Read the dimensions
        dim_match = re.match(rb'^(\d+)\s(\d+)\s$', f.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        # Read the scale
        scale = float(f.readline().rstrip())
        endian = '<' if scale < 0 else '>'  # little or big endian
        scale = abs(scale)

        # Read the data
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # PFM files are stored in top-to-bottom order
        return data, scale

def create_point_cloud(disparity_map, Q):
    """
    Create a point cloud from a disparity map.

    Args:
    - disparity_map: disparity map.
    - Q: 4x4 project matrix.

    Returns:
    - output_points: 3d point clouds.
    """

    # Ensure disparity map is float32
    if disparity_map.dtype != np.float32:
        disparity_map = disparity_map.astype(np.float32)

    # Ensure Q is a 4x4 float32 matrix
    Q = np.array(Q, dtype=np.float32)
    if Q.shape != (4, 4):
        raise ValueError("Q matrix must be a 4x4 matrix.")

    # Reproject points into 3D
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q)

    # Remove points with value 0 (considering them as invalid)
    mask = disparity_map > disparity_map.min()
    output_points = points_3D[mask]

    return output_points

def save_point_cloud(points, filename):
    """
    Save point cloud.

    Args:
    - points: 3d point cloud.
    - filename: output file path.    
    """

    # Create a point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Save to file
    o3d.io.write_point_cloud(filename, point_cloud)

def load_Q_matrix(json_file):
    """
    Load a projection matrix Q from a json file.

    Args:
    - json_file: path to the json file.

    Returns:
    - Q_matrix: projection matrix.    
    """

    with open(json_file, 'r') as file:
        data = json.load(file)
        Q_matrix = np.array(data["reprojection-matrix"])

    return Q_matrix

def display_disparity_maps(ground_truth_map, predicted_map):
    """
    Dispary ground truth and predicted disparity maps side by side.

    Args:
    - gound_truth_map: ground truth disparity map.
    - predicted_map: predicted disparity map.
    """
    
    plt.figure(figsize=(10, 5))

    # Display Ground Truth
    plt.subplot(1, 2, 1)
    plt.imshow(ground_truth_map, cmap='inferno')
    plt.title('Ground Truth Disparity Map')
    plt.colorbar()

    # Display Predicted Map
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_map, cmap='inferno')
    plt.title('Predicted Disparity Map')
    plt.colorbar()

    plt.show()

def calculate_error_metrics(ground_truth, predicted):
    """
    Calculate MSE and MAE between ground truth and prediction.

    Args:
    - ground_truth: ground truth disparity map.
    - predicted: predicted disparity map.

    Returns:
    - mse: Mean Squared Error.
    - mae: Mean Absolute Error.
    """
    mse = np.mean((ground_truth - predicted) ** 2)
    mae = np.mean(np.abs(ground_truth - predicted))
    return mse, mae