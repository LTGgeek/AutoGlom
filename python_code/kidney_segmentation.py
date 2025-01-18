import os
import cv2
import numpy as np
import scipy.io

def create_masked_image(original_img: np.ndarray, mask_img: np.ndarray, background: int) -> np.ndarray:
    """
    Create masked version of image using kidney mask.
    
    Args:
        original_img: Original cropped image
        mask_img: Kidney mask from cvhull directory
        background: Background value (0 for black, 255 for white)
    Returns:
        Masked image with specified background
    """
    # Convert mask to binary (0 and 1)
    binary_mask = (mask_img > 0).astype(np.uint8)
    
    # Create output image with specified background
    output = np.full_like(original_img, background)
    
    # Copy original image values where mask is non-zero
    output = np.where(binary_mask == 1, original_img, output)
    
    return output

def create_binary_mask(mask_img: np.ndarray) -> np.ndarray:
    """
    Convert mask image to binary format.
    
    Args:
        mask_img: Input mask image
    Returns:
        Binary mask as numpy array
    """
    return (mask_img > 0).astype(np.uint8)

def main(target_folderdir: str, sector: str, sslice: int, eslice: int, bdot: bool = False):
    """
    Process kidney images using kidney mask and generate required outputs.
    
    Args:
        target_folderdir: Base directory for all inputs/outputs
        sector: Kidney sector identifier
        sslice: Start slice number
        eslice: End slice number
        bdot: Whether to process bdot masks
    """
    # Define paths
    newpath = os.path.join(target_folderdir, f"{sector}_256")
    
    paths = {
        # Input paths
        'original': os.path.join(newpath, f"{sector}_256_black"),  # Original cropped images
        'kidney_mask': os.path.join(newpath, f"{sector}_kidney_mask_cvhull_256"),  # Kidney masks
        'medulla_mask': os.path.join(newpath, f"{sector}_medulla_mask_cvhull_256"),
        'bdot_mask': os.path.join(newpath, f"{sector}_bdot_mask_cvhull_256"),
        
        # Output paths
        'black_mask': os.path.join(newpath, f"{sector}_kidney_seg_black_mask_256"),
        'white_mask': os.path.join(newpath, f"{sector}_kidney_seg_white_mask_256"),
        'black_array': os.path.join(newpath, f"{sector}_kidney_seg_black_MatArray_256"),
        'white_array': os.path.join(newpath, f"{sector}_kidney_seg_white_MatArray_256"),
        'medulla_array': os.path.join(newpath, f"{sector}_medulla_ann_MatArray_256"),
        'bdot_array': os.path.join(newpath, f"{sector}_bdot_ann_MatArray_256"),
    }

    # Create output directories
    output_dirs = ['black_mask', 'white_mask', 'black_array', 'white_array', 'medulla_array']
    if bdot:
        output_dirs.append('bdot_array')
        
    for key in output_dirs:
        os.makedirs(paths[key], exist_ok=True)

    processed_count = 0
    for slice_num in range(sslice, eslice + 1):
        # Format slice number for filename
        slice_str = f"{slice_num:04d}"
        base_name = f"{sector}{slice_str}"
        
        # Define input paths for current slice
        orig_path = os.path.join(paths['original'], f"{base_name}.png")
        mask_path = os.path.join(paths['kidney_mask'], f"{base_name}.png")
        medulla_path = os.path.join(paths['medulla_mask'], f"{base_name}.png")
        bdot_path = os.path.join(paths['bdot_mask'], f"{base_name}.png")

        # Check if required files exist
        if not (os.path.exists(orig_path) and os.path.exists(mask_path) and os.path.exists(medulla_path) and os.path.exists(bdot_path)):
            print(f"Missing required files for slice {slice_num}")
            continue

        try:
            # Read images
            orig_img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Create and save kidney mask versions
            black_masked = create_masked_image(orig_img, mask_img, background=0)
            white_masked = create_masked_image(orig_img, mask_img, background=255)

            cv2.imwrite(os.path.join(paths['black_mask'], f"{base_name}.png"), black_masked)
            cv2.imwrite(os.path.join(paths['white_mask'], f"{base_name}.png"), white_masked)

            # Save kidney mask arrays
            scipy.io.savemat(os.path.join(paths['black_array'], f"{base_name}.mat"), {'vect': black_masked})
            scipy.io.savemat(os.path.join(paths['white_array'], f"{base_name}.mat"), {'vect': white_masked})

            # Process medulla mask if file exists
            if os.path.exists(medulla_path):
                medulla_img = cv2.imread(medulla_path, cv2.IMREAD_GRAYSCALE)
                medulla_binary = create_binary_mask(medulla_img)
                scipy.io.savemat(os.path.join(paths['medulla_array'], f"{base_name}.mat"), 
                               {'vect': medulla_binary})

            # Process bdot mask if enabled and file exists
            if bdot and os.path.exists(bdot_path):
                bdot_img = cv2.imread(bdot_path, cv2.IMREAD_GRAYSCALE)
                bdot_binary = create_binary_mask(bdot_img)
                scipy.io.savemat(os.path.join(paths['bdot_array'], f"{base_name}.mat"), 
                               {'vect': bdot_binary})

            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} slices...")

        except Exception as e:
            print(f"Error processing slice {slice_num}: {str(e)}")
            continue

    print(f"Processing complete. Successfully processed {processed_count} slices.")
    print("Created files in:")
    for key in output_dirs:
        print(f"- {paths[key]}")

if __name__ == "__main__":
    # Example usage
    SECTOR = "685"
    TARGET_DIR = "d:/ASU_profwu/"
    
    main(
        target_folderdir=TARGET_DIR,
        sector=SECTOR,
        sslice=34,
        eslice=219,
        bdot=True  
    )