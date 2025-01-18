# convert_nrrd_to_png.py

import os
import json
import numpy as np
from utils import (
    read_nrrd,
    select_slice,
    compute_voxel_spacing,
    compute_global_statistics,
    standardize_data,
    sigmoid_scaling,
    apply_threshold,
    save_png,
    compute_bounding_box,
    crop_volume,
    get_annotation_range,
    create_masked_image
)

def create_output_dirs(output_folder: str, scaling_methods: list, apply_thresh: bool):
    """
    Create subdirectories for each scaling method and thresholding within the output folder.
    """
    full_image_dir = os.path.join(output_folder, 'full_image')
    os.makedirs(full_image_dir, exist_ok=True)
    for method in scaling_methods:
        method_dir = os.path.join(full_image_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        if apply_thresh:
            thresh_dir = os.path.join(method_dir, 'thresholded')
            os.makedirs(thresh_dir, exist_ok=True)

    masked_cropped_dir = os.path.join(output_folder, 'masked_cropped')
    os.makedirs(masked_cropped_dir, exist_ok=True)
    
    # Add directories for black and white masks
    mask_types = ['black_mask', 'white_mask']
    for mask_type in mask_types:
        for method in scaling_methods:
            mask_dir = os.path.join(masked_cropped_dir, mask_type, method)
            os.makedirs(mask_dir, exist_ok=True)
            if apply_thresh:
                thresh_dir = os.path.join(mask_dir, 'thresholded')
                os.makedirs(thresh_dir, exist_ok=True)

    # Create directory for annotations
    annotation_dir = os.path.join(masked_cropped_dir, 'annotations')
    os.makedirs(annotation_dir, exist_ok=True)

    # Create directory for subtracted annotations and bdot annotations
    subtracted_annotations_dir = os.path.join(output_folder, 'subtracted_annotations', 'annotations')
    os.makedirs(subtracted_annotations_dir, exist_ok=True)
    bdot_annotations_dir = os.path.join(output_folder, 'bdot_annotations', 'annotations')
    os.makedirs(bdot_annotations_dir, exist_ok=True)


def process_slices(image_data, output_folder, file_name, slice_axis, scaling_methods,
                   apply_thresh, threshold, sigmoid_alpha, subfolder_name, global_mean, global_std, img_name_prefix):
    """
    Generic function to process and save slices with specified scaling methods.
    """
    num_slices = image_data.shape[slice_axis]
    for slice_idx in range(num_slices):
        slice_data = select_slice(image_data, slice_axis, slice_idx)

        for method in scaling_methods:
            if method == 'global_standardization':
                scaled_data = standardize_data(slice_data, global_mean, global_std)
            elif method == 'sigmoid_scaling':
                scaled_data = sigmoid_scaling(slice_data, alpha=sigmoid_alpha)
            else:
                continue  # Skip unknown methods

            output_path = os.path.join(output_folder, subfolder_name, method,
                                       f"{img_name_prefix}{slice_idx:04d}.png")
            save_png(scaled_data, output_path)

            if apply_thresh:
                thresholded_data = apply_threshold(scaled_data, lower_bound=threshold)
                output_path_thresh = os.path.join(output_folder, subfolder_name, method, 'thresholded',
                                                  f"{img_name_prefix}{slice_idx:04d}.png")
                save_png(thresholded_data, output_path_thresh)

        if (slice_idx + 1) % 10 == 0 or (slice_idx + 1) == num_slices:
            print(f"Processed {slice_idx + 1}/{num_slices} slices for {subfolder_name}.")

def process_image_slices(image_data, output_folder, file_name, slice_axis, scaling_methods,
                         apply_thresh, threshold, sigmoid_alpha, img_name_prefix):
    """
    Process and save slices for the full image.
    """
    # global_mean, global_std, _, _ = compute_global_statistics(image_data, slice_axis)
    # process_slices(
    #     image_data=image_data,
    #     output_folder=output_folder,
    #     file_name=file_name,
    #     slice_axis=slice_axis,
    #     scaling_methods=scaling_methods,
    #     apply_thresh=apply_thresh,
    #     threshold=threshold,
    #     sigmoid_alpha=sigmoid_alpha,
    #     subfolder_name='full_image',
    #     global_mean=global_mean,
    #     global_std=global_std,
    #     img_name_prefix= img_name_prefix
    # )

def process_masked_cropped_image(image_data_cropped, annotation_data_cropped, output_folder,
                               file_name, slice_axis, scaling_methods, apply_thresh,
                               threshold, sigmoid_alpha, img_name_prefix):
    """
    Process the masked and cropped image slices with both black and white masks.
    """
    # Create mask directories
    mask_types = ['black_mask', 'white_mask']
    for mask_type in mask_types:
        for method in scaling_methods:
            mask_dir = os.path.join(output_folder, 'masked_cropped', mask_type, method)
            os.makedirs(mask_dir, exist_ok=True)
            if apply_thresh:
                thresh_dir = os.path.join(mask_dir, 'thresholded')
                os.makedirs(thresh_dir, exist_ok=True)

    # Compute global statistics for original image
    global_mean, global_std, _, _ = compute_global_statistics(image_data_cropped, slice_axis)

    # Process each slice
    num_slices = image_data_cropped.shape[slice_axis]
    for slice_idx in range(num_slices):
        # Get the current slice
        slice_data = select_slice(image_data_cropped, slice_axis, slice_idx)
        mask_slice = select_slice(annotation_data_cropped, slice_axis, slice_idx)

        for method in scaling_methods:
            # Scale the data
            if method == 'global_standardization':
                scaled_data = standardize_data(slice_data, global_mean, global_std)
            elif method == 'sigmoid_scaling':
                scaled_data = sigmoid_scaling(slice_data, alpha=sigmoid_alpha)
            else:
                continue

            # Create and save black masked version
            black_masked = create_masked_image(scaled_data, mask_slice, 'black')
            output_path = os.path.join(output_folder, 'masked_cropped', 'black_mask', 
                                     method, f"{img_name_prefix}{slice_idx:04d}.png")
            save_png(black_masked, output_path)

            # Create and save white masked version
            white_masked = create_masked_image(scaled_data, mask_slice, 'white')
            output_path = os.path.join(output_folder, 'masked_cropped', 'white_mask',
                                     method, f"{img_name_prefix}{slice_idx:04d}.png")
            save_png(white_masked, output_path)

            # # Apply thresholding if requested
            # if apply_thresh:
            #     # Black mask with threshold
            #     thresholded_black = apply_threshold(black_masked, lower_bound=threshold)
            #     output_path = os.path.join(output_folder, 'masked_cropped', 'black_mask',
            #                              method, 'thresholded', f"{img_name_prefix}{slice_idx:04d}.png")
            #     save_png(thresholded_black, output_path)

            #     # White mask with threshold
            #     thresholded_white = apply_threshold(white_masked, lower_bound=threshold)
            #     output_path = os.path.join(output_folder, 'masked_cropped', 'white_mask',
            #                              method, 'thresholded', f"{img_name_prefix}{slice_idx:04d}.png")
            #     save_png(thresholded_white, output_path)

        if (slice_idx + 1) % 10 == 0 or (slice_idx + 1) == num_slices:
            print(f"Processed {slice_idx + 1}/{num_slices} masked slices.")


def save_cropped_annotations(annotation_data, output_folder, slice_axis, subfolder_name, img_name_prefix):
    """
    Save cropped annotation slices.
    """
    annotation_output_dir = os.path.join(output_folder, subfolder_name, 'annotations')
    os.makedirs(annotation_output_dir, exist_ok=True)
    num_slices = annotation_data.shape[slice_axis]

    for slice_idx in range(num_slices):
        slice_data = select_slice(annotation_data, slice_axis, slice_idx)
        binary_slice = np.where(slice_data > 0, 255, 0).astype(np.uint8)
        output_path = os.path.join(annotation_output_dir, f"{img_name_prefix}{slice_idx:04d}.png")
        save_png(binary_slice, output_path)

    print(f"Saved {subfolder_name} annotation PNGs to {annotation_output_dir}")

def generate_metadata(image_file, annotation_file1, annotation_file2, bdot_file, voxel_spacing,
                      slice_axis, sigmoid_alpha, threshold, num_slices_full,
                      num_slices_masked, bounding_box, output_folder, kidney_range, medula_range, bdot_range=None):
    """
    Generate and save metadata to a JSON file.
    """
    metadata = {
        'input_image_file': os.path.abspath(image_file),
        'input_annotation_file1': os.path.abspath(annotation_file1),
        'input_annotation_file2': os.path.abspath(annotation_file2) if annotation_file2 else None,
        'input_bdot_file': os.path.abspath(bdot_file) if bdot_file else None,
        'voxel_spacing': {
            'x': voxel_spacing[0],
            'y': voxel_spacing[1],
            'z': voxel_spacing[2]
        },
        'slice_axis': slice_axis,
        'sigmoid_alpha': sigmoid_alpha,
        'threshold': threshold,
        'total_slices_full_image': num_slices_full,
        'total_slices_masked_cropped': num_slices_masked,
        'adjusted_bounding_box': {
            'min_x': int(bounding_box[0]),
            'max_x': int(bounding_box[1]),
            'min_y': int(bounding_box[2]),
            'max_y': int(bounding_box[3]),
            'min_z': int(bounding_box[4]),
            'max_z': int(bounding_box[5]),
            'pad_x': int(bounding_box[6]),
            'pad_y': int(bounding_box[7]),
            'pad_z': int(bounding_box[8])
        },
        'kidney_range': {
            'min': kidney_range[0],
            'max': kidney_range[1]
        },
        'medula_range': {
            'min': medula_range[0],
            'max': medula_range[1]
        }
    }

    if bdot_range is not None:
        metadata['bdot_range'] = {
            'min': bdot_range[0],
            'max': bdot_range[1]
        }

    metadata_path = os.path.join(output_folder, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved metadata to {metadata_path}")

def convert_nrrd_to_png(image_file, annotation_file1, annotation_file2, bdot_file=None, output_folder=None,
                        slice_axis=1, sigmoid_alpha=0.1, threshold=None, scaling_method='global_standardization', img_name_prefix="100"):
    """
    Orchestrates the conversion of NRRD to PNGs with optional cropping and masking.
    """
    scaling_methods = [scaling_method]
    apply_thresh = threshold is not None

    try:
        print(f"Reading image NRRD file: {image_file}")
        image_data, image_header = read_nrrd(image_file)
        if image_data.ndim != 3:
            raise ValueError(f"Error: The input file {image_file} is not a 3D NRRD file.")
    except Exception as e:
        raise IOError(f"Failed to read image NRRD file {image_file}: {e}")

    try:
        print(f"Reading first annotation NRRD file: {annotation_file1}")
        annotation_data1, _ = read_nrrd(annotation_file1)
        if annotation_data1.shape != image_data.shape:
            raise ValueError("Error: Image and first annotation volumes must have the same shape.")
    except Exception as e:
        raise IOError(f"Failed to read annotation NRRD file {annotation_file1}: {e}")

    # Read the second annotation file if provided
    annotation_data2 = None
    if annotation_file2:
        try:
            print(f"Reading second annotation NRRD file: {annotation_file2}")
            annotation_data2, _ = read_nrrd(annotation_file2)
            if annotation_data2.shape != image_data.shape:
                raise ValueError("Error: Image and second annotation volumes must have the same shape.")
        except Exception as e:
            raise IOError(f"Failed to read annotation NRRD file {annotation_file2}: {e}")

    # Read the bdot file if provided
    bdot_data = None
    bdot_range = None
    if bdot_file:
        try:
            print(f"Reading bdot NRRD file: {bdot_file}")
            bdot_data, _ = read_nrrd(bdot_file)
            if bdot_data.shape != image_data.shape:
                raise ValueError("Error: Image and bdot volumes must have the same shape.")
        except Exception as e:
            raise IOError(f"Failed to read bdot NRRD file {bdot_file}: {e}")

    voxel_spacing = compute_voxel_spacing(image_header)
    print(f"Voxel Spacing (x, y, z): {voxel_spacing}")

    file_name = os.path.splitext(os.path.basename(image_file))[0]
    create_output_dirs(output_folder, scaling_methods, apply_thresh)

    # print("\nProcessing full image...")
    # process_image_slices(image_data, output_folder, file_name, slice_axis, scaling_methods,
    #                      apply_thresh, threshold, sigmoid_alpha, img_name_prefix)

    # Compute the bounding box from annotation_data1
    bounding_box = compute_bounding_box(annotation_data1, slice_axis)

    # Crop image_data and annotation_data1 using the bounding box
    image_data_cropped = crop_volume(image_data, bounding_box)
    annotation_data1_cropped = crop_volume(annotation_data1, bounding_box)

    print("\nProcessing masked and cropped image using the first annotation...")
    process_masked_cropped_image(image_data_cropped, annotation_data1_cropped, output_folder,
                                 file_name, slice_axis, scaling_methods, apply_thresh,
                                 threshold, sigmoid_alpha, img_name_prefix)

    # Save the cropped annotations from annotation_data1
    save_cropped_annotations(annotation_data1_cropped, output_folder, slice_axis, subfolder_name='masked_cropped', img_name_prefix=img_name_prefix)

    # Calculate kidney range
    kidney_range = get_annotation_range(annotation_data1, slice_axis)
    print(f"Kidney Range: {kidney_range}")

    # Process annotation_data2 if provided
    medula_range = None
    if annotation_data2 is not None:
        print("\nSubtracting second annotation from the first...")
        annotation_data2_cropped = crop_volume(annotation_data2, bounding_box)
        annotation_data1_subtracted = np.logical_and(annotation_data1_cropped > 0, annotation_data2_cropped == 0).astype(np.uint8)
        print("Saving subtracted annotations...")
        save_cropped_annotations(annotation_data1_subtracted, output_folder, slice_axis, subfolder_name='subtracted_annotations', img_name_prefix=img_name_prefix)
        save_cropped_annotations(annotation_data2_cropped, output_folder, slice_axis, subfolder_name='medulla_annotations', img_name_prefix=img_name_prefix)
        medula_range = get_annotation_range(annotation_data2_cropped, slice_axis)
        print(f"Medulla Range: {medula_range}")

    # Process bdot_data if provided
    if bdot_data is not None:
        print("\nProcessing bdot annotations...")
        bdot_data_cropped = crop_volume(bdot_data, bounding_box)
        save_cropped_annotations(bdot_data_cropped, output_folder, slice_axis, subfolder_name='bdot_annotations', img_name_prefix=img_name_prefix)
        bdot_range = get_annotation_range(bdot_data_cropped, slice_axis)
        print(f"Bdot Range: {bdot_range}")
        if annotation_data2 is not None:
            print("\nSubtracting second annotation from the first...")
            annotation_data2_cropped = crop_volume(annotation_data2, bounding_box)
            annotation_data1_subtracted = np.logical_and(
                annotation_data1_cropped > 0,
                np.logical_and(annotation_data2_cropped == 0, bdot_data_cropped == 0)
            ).astype(np.uint8)
            print("Saving subtracted annotations...")
            save_cropped_annotations(annotation_data1_subtracted, output_folder, slice_axis, subfolder_name='subtracted_annotations', img_name_prefix=img_name_prefix)
            save_cropped_annotations(annotation_data2_cropped, output_folder, slice_axis, subfolder_name='medulla_annotations', img_name_prefix=img_name_prefix)
            medula_range = get_annotation_range(annotation_data2_cropped, slice_axis)
            print(f"Medulla Range: {medula_range}")
    

    num_slices_full = image_data.shape[slice_axis]
    num_slices_masked = annotation_data1_cropped.shape[slice_axis]

    generate_metadata(image_file, annotation_file1, annotation_file2, bdot_file, voxel_spacing,
                      slice_axis, sigmoid_alpha, threshold, num_slices_full, num_slices_masked,
                      bounding_box, output_folder, kidney_range, medula_range, bdot_range)
 

def parse_arguments():
    """
    Parse command line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Convert a 3D NRRD image and annotations to PNG images with and without cropping.")
    parser.add_argument('--image_file', type=str, required=True, help='Path to the input 3D NRRD image file.')
    parser.add_argument('--annotation_file1', type=str, required=True, help='Path to the input 3D NRRD annotation file (e.g., kidney).')
    parser.add_argument('--annotation_file2', type=str, default=None, help='(Optional) Path to a second input 3D NRRD annotation file to subtract.')
    parser.add_argument('--bdot_file', type=str, default=None, help='(Optional) Path to the bdot NRRD file.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the folder to save PNG images.')
    parser.add_argument('--slice_axis', type=int, default=2, choices=[0, 1, 2], help='Axis along which to slice the 3D data (0 for X, 1 for Y, 2 for Z). Default is 2.')
    parser.add_argument('--sigmoid_alpha', type=float, default=0.1, help='Alpha value for sigmoid scaling. Controls steepness. Default is 0.1.')
    parser.add_argument('--threshold', type=int, default=None, help='Lower bound threshold (0-255) to reduce noise. Optional.')
    return parser.parse_args()

# If the script is run directly, execute the main function
if __name__ == "__main__":
    args = parse_arguments()
    convert_nrrd_to_png(
        image_file=args.image_file,
        annotation_file1=args.annotation_file1,
        annotation_file2=args.annotation_file2,  
        bdot_file=args.bdot_file,  # Optional; can be None
        output_folder=args.output_folder,
        slice_axis=args.slice_axis,
        sigmoid_alpha=args.sigmoid_alpha,
        threshold=args.threshold
    )
