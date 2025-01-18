import os
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from scipy.ndimage import label
import json
from .utils import *

structure_26_connectivity = np.ones((3, 3, 3))

def run_uhdog(config):
    """
    UH-DoG algorithm for glomeruli segmentation, Python implementation
    
    Args:
        config: Dictionary containing configuration parameters
    """
    # Extract parameters from config
    folder_path = config['folder_path']
    print("folder_path:", folder_path)
    print(folder_path)
    kidney = config['kidney']
    sslice = config['sslice']
    eslice = config['eslice']
    med_start_slice = config['med_start_slice']
    med_end_slice = config['med_end_slice']
    dist_boundary = config['dist_boundary']
    inten_thre = config['inten_thre']
    perct = config['perct']
    unet_mask_threshold = config['unet_mask_threshold']
    x_space = config['x_space']
    y_space = config['y_space']
    z_space = config['z_space']
    use_blackdot_mask = config['use_blackdot_mask']

    # Construct base variables
    kidney_id = kidney
    folder_256 = f'{kidney_id}_256'
    seg_array_name = f'{kidney_id}_256_black'
    unet_output = f'{kidney_id}_output'

    # Full paths for various files
    kimg_path = os.path.join(folder_path, folder_256, seg_array_name)
    pimg_path = os.path.join(folder_path, unet_output)

    # Mask paths
    medulla_mask_folder = f'{kidney_id}_medulla_mask_cvhull_256'
    blackdot_mask_folder = f'{kidney_id}_bdot_mask_cvhull_256'
    kidney_mask_folder = f'{kidney_id}_kidney_mask_cvhull_256'

    medulla_mask_path = os.path.join(folder_path, folder_256, medulla_mask_folder)
    blackdot_mask_path = os.path.join(folder_path, folder_256, blackdot_mask_folder)
    kidney_mask_path = os.path.join(folder_path, folder_256, kidney_mask_folder)

    
    # Results paths
    results_dir = f'{kidney_id}_results'
    save_dir = os.path.join(folder_path, results_dir)
    # save_path = os.path.join(save_dir, 'results.npz')
    save_json_path = os.path.join(save_dir, 'results.json')
    
    # Create results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    kimg = load_images_to_3d_array(kimg_path)
    pimg = load_unet_images_to_3d_array(pimg_path)


    imgb = 1 - kimg.astype(bool)  # Convert to binary and invert
    bw = binary_fill_holes(1 - imgb).astype(int)  # Fill holes

    # Compute distance transform
    dist = distance_transform_edt(bw)

    # Process PIMG based on distance threshold
    pimg2 = pimg.copy()
    pimg2[dist < dist_boundary] = 1

    # Create the mask by inverting PIMG2
    pimg_mask = 1 - pimg2

    # Apply thresholding
    pimg_mask[pimg_mask < unet_mask_threshold] = 0
    pimg_mask[pimg_mask >= unet_mask_threshold] = 1

    # Generate Hessian convexity map from HDoG
    log, h_mask = dog_search(kimg, 'negative', sigma=1)

    # Apply thresholding
    intensity, inten_var = inten_extraction(kimg, h_mask, 6)

    # Assuming Intensity, intenThre, and HMask are already defined
    indx = intensity > inten_thre  # Apply threshold
    r = np.where(indx)[0]  # Find indices where Intensity > intenThre

    # Create a boolean mask for HMask based on these indices
    it = np.isin(h_mask.ravel(), r + 1)  # +1 because labels are 1-based in MATLAB, 0-based in Python
    post_mask = it.reshape(h_mask.shape)  # Reshape to original shape of HMask

    mmask = load_annotations_to_3d_array(medulla_mask_path)
    kmask = load_annotations_to_3d_array(kidney_mask_path)


    # 8. Apply masks
    if use_blackdot_mask:
        bdot_mask = load_annotations_to_3d_array(blackdot_mask_path)
        uh_dog_mask = h_mask * pimg_mask * post_mask * (1 - mmask) * (1 - bdot_mask)
    else:
        uh_dog_mask = h_mask * pimg_mask * post_mask * (1 - mmask)

    label_mask, n_glom = label(uh_dog_mask, structure=structure_26_connectivity)

    BMask = (label_mask > 0).astype(int)

    # Generate the overlay image
    overlay_img = overlay_red_dots(kimg, BMask)
    save_rgb_slices_as_png(overlay_img, save_dir)

    # Calculate volumes for each region in LabelMask
    vs = get_all_vol(kimg, label_mask, perct, x_space, y_space, z_space)

    # Calculate mean and median of non-zero volumes
    mean_vs = np.mean(vs[vs > 0])
    med_vs = np.median(vs[vs > 0])

    _,kid_volume = kidney_volume(kmask, x_space, y_space, z_space)
    _,med_volume = kidney_volume(mmask, x_space, y_space, z_space)
    if use_blackdot_mask:
        _,bd_volume = kidney_volume(bdot_mask, x_space, y_space, z_space)

    json_data = {
        'n_glom': int(n_glom),
        'mean_vs': float(mean_vs),
        'med_vs': float(med_vs),
        'start_slice': int(sslice),
        'end_slice': int(eslice),
        'inten_thre': float(inten_thre),
        'use_blackdot_mask': bool(use_blackdot_mask),
        'kidney_volume': float(kid_volume),
        'medulla_volume': float(med_volume),
        'unit' : 'mm^3',
    }

    if use_blackdot_mask:
        json_data['blackdot_volume'] = float(bd_volume)

    with open(save_json_path, 'w') as f:
        json.dump(json_data, f, indent=4)


