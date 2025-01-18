# utils.py

import numpy as np
from typing import Tuple
from PIL import Image
import nrrd
import os

def read_nrrd(file_path: str) -> Tuple[np.ndarray, dict]:
    """
    Reads a NRRD file and returns the data as a NumPy array and the header.
    """
    try:
        data, header = nrrd.read(file_path)
        return data, header
    except Exception as e:
        raise IOError(f"Failed to read NRRD file {file_path}: {e}")

def select_slice(data: np.ndarray, axis: int, index: int) -> np.ndarray:
    """
    Selects a slice from the 3D data along the specified axis and index.
    """
    if axis == 0:
        return data[index, :, :]
    elif axis == 1:
        return data[:, index, :]
    elif axis == 2:
        return data[:, :, index]
    else:
        raise ValueError("Invalid axis value. Must be 0, 1, or 2.")

def compute_voxel_spacing(header: dict) -> Tuple[float, float, float]:
    """
    Extracts voxel spacing (spacing_x, spacing_y, spacing_z) from the NRRD header.
    """
    if 'space directions' in header:
        space_directions = header['space directions']
        spacing_x = np.linalg.norm(space_directions[0])
        spacing_y = np.linalg.norm(space_directions[1])
        spacing_z = np.linalg.norm(space_directions[2])
        return spacing_x, spacing_y, spacing_z
    elif 'spacings' in header:
        spacings = header['spacings']
        if len(spacings) != 3:
            raise ValueError("Expected three spacing values for a 3D image.")
        return tuple(spacings)
    else:
        raise KeyError("Voxel spacing information not found in the NRRD header.")

def compute_global_statistics(data: np.ndarray, axis: int = 2) -> Tuple[float, float, float, float]:
    """
    Compute global statistics (mean, std, median, IQR) across all slices.
    """
    flattened_data = data.flatten()
    global_mean = np.mean(flattened_data)
    global_std = np.std(flattened_data)
    global_median = np.median(flattened_data)
    q25, q75 = np.percentile(flattened_data, [25, 75])
    global_iqr = q75 - q25
    return global_mean, global_std, global_median, global_iqr

def standardize_data(data: np.ndarray, global_mean: float, global_std: float) -> np.ndarray:
    """
    Standardize data using global mean and standard deviation.
    Scales to 0-255 without clipping.
    """
    standardized = (data - global_mean) / global_std
    min_val = standardized.min()
    max_val = standardized.max()
    normalized = (standardized - min_val) / (max_val - min_val)
    scaled = (normalized * 255).astype(np.uint8)
    return scaled

def sigmoid_scaling(data: np.ndarray, alpha: float = 0.1, beta: float = None) -> np.ndarray:
    """
    Apply sigmoid function for smooth scaling.
    """
    if beta is None:
        beta = np.median(data)
    scaled = 1 / (1 + np.exp(-alpha * (data - beta)))
    scaled = (scaled * 255).astype(np.uint8)
    return scaled

def apply_threshold(slice_data: np.ndarray, lower_bound: int = 0) -> np.ndarray:
    """
    Apply thresholding to the 8-bit image to reduce noise.
    Pixels below the lower_bound are set to 0.
    """
    if not (0 <= lower_bound <= 255):
        raise ValueError("lower_bound must be in the range [0, 255].")
    thresholded = np.where(slice_data < lower_bound, 0, slice_data)
    return thresholded.astype(np.uint8)

def save_png(slice_data: np.ndarray, output_path: str):
    """
    Save a 2D NumPy array as a PNG image.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image = Image.fromarray(slice_data)
    image.save(output_path)

def compute_bounding_box(annotation_data: np.ndarray, axis: int) -> Tuple[int, int, int, int, int, int, int, int, int]:
    """
    Computes the original bounding box around non-zero regions and provides padding values
    based on the specified axis.
    
    Args:
        annotation_data: 3D numpy array containing the annotation data
        axis: Determines which dimension stays unchanged while the other two are matched:
              axis=0: x dimension unchanged, y and z are padded to match each other
              axis=1: y dimension unchanged, x and z are padded to match each other
              axis=2: z dimension unchanged, x and y are padded to match each other
    
    Returns: min_x, max_x, min_y, max_y, min_z, max_z, pad_x, pad_y, pad_z
    For each axis value, one padding will be 0 (the unchanged dimension) while
    the other two will be padded to match the larger of those two dimensions.
    """
    # Find non-zero indices
    non_zero_indices = np.argwhere(annotation_data)
    if non_zero_indices.size == 0:
        print("Warning: No non-zero elements found in annotation data.")
        return (0, annotation_data.shape[0] - 1, 0, annotation_data.shape[1] - 1,
                0, annotation_data.shape[2] - 1, 0, 0, 0)

    # Compute original bounding box
    min_coords = non_zero_indices.min(axis=0)
    max_coords = non_zero_indices.max(axis=0)
    min_x, min_y, min_z = min_coords
    max_x, max_y, max_z = max_coords

    # Calculate dimensions
    width = max_x - min_x + 1   # x dimension
    height = max_y - min_y + 1  # y dimension
    depth = max_z - min_z + 1   # z dimension

    print(f"Original bounding box: ({min_x}, {min_y}, {min_z}), ({max_x}, {max_y}, {max_z})")

    if axis == 0:
        # For axis=0, x remains unchanged, make y and z equal to the larger of the two
        target_size = max(height, depth)
        pad_x = 0  # x dimension unchanged
        pad_y = target_size - height if height < target_size else 0
        pad_z = target_size - depth if depth < target_size else 0
        print(f"Padding needed - x: {pad_x}, y: {pad_y}, z: {pad_z}")
        
    elif axis == 1:
        # For axis=1, y remains unchanged, make x and z equal to the larger of the two
        target_size = max(width, depth)
        pad_x = target_size - width if width < target_size else 0
        pad_y = 0  # y dimension unchanged
        pad_z = target_size - depth if depth < target_size else 0
        print(f"Padding needed - x: {pad_x}, y: {pad_y}, z: {pad_z}")
        
    elif axis == 2:
        # For axis=2, z remains unchanged, make x and y equal to the larger of the two
        target_size = max(width, height)
        pad_x = target_size - width if width < target_size else 0
        pad_y = target_size - height if height < target_size else 0
        pad_z = 0  # z dimension unchanged
        print(f"Padding needed - x: {pad_x}, y: {pad_y}, z: {pad_z}")
        
    else:
        raise ValueError(f"Invalid axis value: {axis}. Must be 0, 1, or 2.")

    return (int(min_x), int(max_x), int(min_y), int(max_y), int(min_z), int(max_z),
            int(pad_x), int(pad_y), int(pad_z))


def crop_volume(volume: np.ndarray, bounding_box: Tuple[int, int, int, int, int, int, int, int, int], 
                pad_value: float = 0) -> np.ndarray:
    """
    Crops the volume using the provided bounding box and applies padding if specified.
    Args:
        volume: Input 3D volume
        bounding_box: Tuple of (min_x, max_x, min_y, max_y, min_z, max_z, pad_x, pad_y)
        pad_value: Value to use for padding (default=0)
    Returns:
        Cropped and padded volume
    """
    min_x, max_x, min_y, max_y, min_z, max_z, pad_x, pad_y, pad_z = bounding_box
    
    # Initial crop
    cropped = volume[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
    
    # Apply padding if needed
    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        print(f"Padding with value {pad_value} ({pad_x}, {pad_y})")
        pad_width = ((0, pad_x), (0, pad_y), (0, pad_z))
        cropped = np.pad(cropped, pad_width, mode='constant', constant_values=pad_value)
    
    print(f"Cropped volume shape: {cropped.shape}")
    
    return cropped

def get_annotation_range(annotation_data: np.ndarray, axis: int = 2) -> Tuple[int, int]:
    """
    Find the first and last slices containing non-zero annotations along the specified axis.
    
    Args:
        annotation_data: 3D numpy array containing the annotation data
        axis: Axis along which to find the range (0 for X, 1 for Y, 2 for Z)
    
    Returns:
        Tuple[int, int]: (start_slice, end_slice) indices where annotations are present
    """
    # Sum along all axes except the specified one to get a 1D array
    axes_to_sum = tuple(i for i in range(3) if i != axis)
    slice_sums = np.sum(annotation_data, axis=axes_to_sum)
    
    # Find indices where the sum is non-zero
    non_zero_indices = np.nonzero(slice_sums)[0]
    
    if len(non_zero_indices) == 0:
        raise ValueError("No annotations found in the volume")
    
    start_slice = int(non_zero_indices[0])
    end_slice = int(non_zero_indices[-1])
    
    return start_slice, end_slice
def create_masked_image(slice_data: np.ndarray, mask: np.ndarray, mask_color: str = 'black') -> np.ndarray:
    """
    Create a masked version of the image with either black or white mask.
    
    Args:
        slice_data: 2D array of the image slice
        mask: Binary mask (same shape as slice_data)
        mask_color: 'black' or 'white' for mask color
    
    Returns:
        Masked image as uint8 array
    """
    # Make sure both arrays are the same shape
    if slice_data.shape != mask.shape:
        raise ValueError("Image and mask must have the same shape")
        
    # Create binary mask
    binary_mask = (mask > 0)
    
    # Create masked image
    if mask_color.lower() == 'black':
        masked_image = np.where(~binary_mask, 0, slice_data)
    elif mask_color.lower() == 'white':
        masked_image = np.where(~binary_mask, 255, slice_data)
    else:
        raise ValueError("mask_color must be either 'black' or 'white'")
        
    return masked_image.astype(np.uint8)