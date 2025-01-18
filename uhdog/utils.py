import os
import numpy as np
from skimage.io import imread
from skimage import img_as_float
from scipy.ndimage import convolve
from scipy.ndimage import label
from skimage.transform import resize
from scipy.io import loadmat
from PIL import Image

def squared_gaussian(sigma, size=None):
    if size is None:
        size = [int(np.ceil(3 * sigma))] * 3  # Set size based on sigma if not provided

    # Create a 3D grid with ndgrid equivalent
    x, y, z = np.meshgrid(
        np.arange(-size[0], size[0] + 1),
        np.arange(-size[1], size[1] + 1),
        np.arange(-size[2], size[2] + 1),
        indexing='ij'
    )
    
    # Compute the squared Gaussian
    H = np.exp(-(x**2 / (2 * sigma**2) + y**2 / (2 * sigma**2) + z**2 / (2 * sigma**2)))
    H /= H.sum()  # Normalize H so that the sum of all elements is 1
    
    return H

def gaussian_transformer(IMG, sigma):
    H = squared_gaussian(sigma)  # Generate the Gaussian kernel
    #print(f'Transforming to Gaussian space with sigma={sigma}')
    # Perform convolution using 'reflect' mode for boundary handling, similar to 'same' in MATLAB
    IMG = convolve(IMG, H, mode='reflect')
    
    return IMG


def load_images_to_3d_array(directory):
    """
    Loads all .png images in the specified directory into a 3D numpy array.

    Parameters:
    directory (str): Path to the directory containing the .png images.

    Returns:
    np.ndarray: A 3D numpy array where each slice along the third axis is an image.
    """
    # List of all .png files in the directory
    namelist = sorted([f for f in os.listdir(directory) if f.endswith('.png')])
    len_files = len(namelist)
    
    # Check if the directory contains any PNG files
    if len_files == 0:
        raise ValueError("No .png files found in the specified directory.")
    
    # Read the first image to get its dimensions
    first_image_path = os.path.join(directory, namelist[0])
    first_image = imread(first_image_path)
    image_shape = first_image.shape  # Get the (height, width) of the first image
    
    # Initialize an empty array to store the images based on the first image's dimensions
    scan_img = np.zeros((image_shape[0], image_shape[1], len_files), dtype=float)
    
    # Loop through each image file, read it, convert to float, and store in cart_img
    for i, file_name in enumerate(namelist):
        file_path = os.path.join(directory, file_name)
        tmp = imread(file_path)
        tmp = img_as_float(tmp)  # Converts to double (float) in the range [0, 1]
        scan_img[:, :, i] = tmp
    
    return scan_img

def load_annotations_to_3d_array(directory):
    """
    Loads all .png images in the specified directory into a 3D numpy array.

    Parameters:
    directory (str): Path to the directory containing the .png images.

    Returns:
    np.ndarray: A 3D numpy array where each slice along the third axis is an image.
    """
    # List of all .png files in the directory
    namelist = sorted([f for f in os.listdir(directory) if f.endswith('.png')])
    len_files = len(namelist)
    
    # Check if the directory contains any PNG files
    if len_files == 0:
        raise ValueError("No .png files found in the specified directory.")
    
    # Read the first image to get its dimensions
    first_image_path = os.path.join(directory, namelist[0])
    first_image = imread(first_image_path)
    image_shape = first_image.shape  # Get the (height, width) of the first image
    
    # Initialize an empty array to store the images based on the first image's dimensions
    scan_ann = np.zeros((image_shape[0], image_shape[1], len_files), dtype=float)
    
    # Loop through each image file, read it, convert to float, and store in cart_img
    for i, file_name in enumerate(namelist):
        file_path = os.path.join(directory, file_name)
        tmp = imread(file_path)
        tmp = img_as_float(tmp)  # Converts to double (float) in the range [0, 1]
        scan_ann[:, :, i] = tmp

    scan_ann[scan_ann>=0.1] = 1
    scan_ann[scan_ann<1] = 0
    
    return scan_ann

def load_unet_images_to_3d_array(directory):
    """
    Loads all .mat files in the specified directory into a 3D numpy array.
    Expects each .mat file to contain a 'vect' variable with the image data.

    Parameters:
    directory (str): Path to the directory containing the .mat files.

    Returns:
    np.ndarray: A 3D numpy array where each slice along the third axis is an image.
            Values are normalized to the range [0, 1].
    """
    # List of all .mat files in the directory
    namelist = sorted([f for f in os.listdir(directory) if f.endswith('.mat')])
    len_files = len(namelist)
    
    # Check if the directory contains any .mat files
    if len_files == 0:
        raise ValueError("No .mat files found in the specified directory.")
    
    # Read the first image to get its dimensions
    first_image_path = os.path.join(directory, namelist[0])
    first_mat = loadmat(first_image_path)
    first_image = first_mat['vect']
    image_shape = first_image.shape  # Get the (height, width) of the first image
    
    # Initialize an empty array to store the images
    unet_img = np.zeros((image_shape[0], image_shape[1], len_files), dtype=float)
    
    # Loop through each .mat file and store in unet_img
    for i, file_name in enumerate(namelist):
        file_path = os.path.join(directory, file_name)
        mat_data = loadmat(file_path)
        unet_img[:, :, i] = mat_data['vect']
    
    # Normalize the entire array to [0, 1]
    if unet_img.size > 0:
        unet_img = (unet_img - unet_img.min()) / (unet_img.max() - unet_img.min())
    
    return unet_img


def convex_detector(IMG, type_):

    # Define the structure for 6-connectivity
    structure_6_connectivity = np.array([[[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]],
                      
                      [[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]],
                      
                      [[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]]], dtype=int)
    
    nx, ny, nz = IMG.shape
    
    # Calculate gradients
    dx, dy, dz = np.gradient(IMG)
    dxx, dxy, dxz = np.gradient(dx)
    _, dyy, dyz = np.gradient(dy)
    _, _, dzz = np.gradient(dz)
    
    # Initialize U, V, Q matrices
    U = np.zeros((nx, ny, nz))
    V = dxx * dyy - dxy * dxy
    Q = (dxx * dyy * dzz + 2 * dxy * dyz * dxz) - (dxz * dxz * dyy + dxy * dxy * dzz + dyz * dyz * dxx)
    
    # Apply the conditions based on type
    if type_ == 'positive':
        U[dxx > 0] = 1
        V = np.where(V > 0, 1, 0)
        Q = np.where(Q > 0, 1, 0)
    elif type_ == 'semi-positive':
        U[dxx >= 0] = 1
        V = np.where(V >= 0, 1, 0)
        Q = np.where(Q >= 0, 1, 0)
    elif type_ == 'negative':
        U[dxx < 0] = 1
        V = np.where(V > 0, 1, 0)
        Q = np.where(Q > 0, 0, 1)
    elif type_ == 'semi-negative':
        U[dxx <= 0] = 1
        V = np.where(V >= 0, 1, 0)
        Q = np.where(Q > 0, 2, 1)
    else:
        raise ValueError('Unknown filter type.')
    
    # Combine U, V, Q to create Mask
    MaskO = U + Q + V
    Mask = np.copy(MaskO)
    Mask[Mask < 3] = 0
    Mask[Mask == 3] = 1
    
    # Compute LoG
    if np.sum(Mask) > 0:
        LoG = np.sum(IMG * Mask) / np.sum(Mask)
    else:
        LoG = 0  # Avoid division by zero if Mask is empty
    
    # Label connected components in Mask
    Mask, num_features = label(Mask, structure=structure_6_connectivity)  # Connectivity of 6
    
    return Mask, LoG

def dog_search(IMG, type_, sigma=None):
    low = 0.8
    up = 1.5
    num_interval = (up - low) / 0.1

    if sigma is None:
        print("Searching best DoG Space ...")
        t = 1
        maxt = 0
        maxi = 0
        T = []
        
        for sigma in np.arange(low, up + 0.1, 0.1):
            TLOG = sigma ** (2 - 1) * (gaussian_transformer(IMG, sigma + 0.001) - gaussian_transformer(IMG, sigma)) / 0.001
            Temp, T_val = convex_detector(TLOG, type_)
            T.append(T_val)
            
            if T_val > maxi:
                print(maxi)
                print(T_val)
                LOG = TLOG
                Mask = Temp
                maxt = sigma
                maxi = T_val
            
            t += 1
        
        print(T)
        print(f"Optimum Space with sigma={maxt}")
    
    elif sigma is not None:
        LOG = sigma ** (2 - 1) * (gaussian_transformer(IMG, sigma + 0.001) - gaussian_transformer(IMG, sigma)) / 0.001
        Mask, T_val = convex_detector(LOG, type_)
    
    return LOG, Mask

def inten_extraction(IMG, Mask, ConnNum):
    """
    Extracts the mean and variance of intensity values within labeled regions.
    
    Parameters:
    IMG (ndarray): The intensity image.
    Mask (ndarray): The binary mask defining regions of interest.
    ConnNum (int): Connectivity number (6 or 26 for 3D).
    
    Returns:
    tuple: (Intensity, IntenVar) where Intensity is the mean intensity per region,
           and IntenVar is the variance of intensity per region.
    """
    # Define connectivity structure for labeling
    if ConnNum == 6:
        structure = np.array([[[0, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]],
                              [[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]],
                              [[0, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]]], dtype=int)
    elif ConnNum == 26:
        structure = np.ones((3, 3, 3), dtype=int)
    else:
        raise ValueError("ConnNum must be 6 or 26 for 3D connectivity.")
    
    # Label connected components in the mask
    L, num_features = label(Mask, structure=structure)
    
    # Find indices of labeled regions
    infd = np.where(L > 0)
    LL = L[infd]
    Int = IMG[infd].flatten()

    # Calculate mean intensity for each labeled region
    Intensity = np.zeros(num_features)
    IntenVar = np.zeros(num_features)
    
    for i in range(1, num_features + 1):
        region_intensities = Int[LL == i]
        Intensity[i - 1] = region_intensities.mean()
        IntenVar[i - 1] = region_intensities.var()
    
    return Intensity, IntenVar

def overlay_red_dots(kimg, BMask):
    """
    Overlays BMask as red dots on kimg.
    
    Parameters:
    kimg (numpy.ndarray): The grayscale image.
    BMask (numpy.ndarray): Binary mask indicating where to place red dots.
    
    Returns:
    numpy.ndarray: RGB image with red dots overlaid on kimg.
    """
    # Normalize and scale kimg to 0-255 if necessary, then convert to uint8
    if kimg.max() <= 1.0:
        kimg = (kimg * 255).astype(np.uint8)
    
    # Create an RGB version of kimg
    kimg_rgb = np.stack([kimg, kimg, kimg], axis=-1)

    # Overlay red dots: set red channel to 255 where BMask is 1, keep green and blue at 0
    kimg_rgb[BMask > 0] = [255, 0, 0]
    
    return kimg_rgb



def getllength(line, perct):
    """
    Calculates the diameter based on intensity values in `line` and a percentage threshold `perct`.
    
    Parameters:
    line (ndarray): 1D array of intensity values.
    perct (float): Percentage threshold for calculating height adjustment.
    
    Returns:
    float: Calculated diameter or `total`.
    """
    # Normalize the line by subtracting the maximum value
    line = line - np.max(line)

    # Calculate the height and adjust line values by the threshold percentage
    height = abs(np.min(line))
    line = line + perct * height

    llen = len(line)
    mid1 = (llen + 1) / 2
    mid2 = np.argmin(line)
    mid = mid2 if abs(mid2 - mid1) <= 2 else int(mid1)

    dline = np.diff(line)

    # Find left limit (li)
    for i in range(mid, -1, -1):
        if dline[i - 1] >= -0.01 and line[i] > -0.1 * abs(line[mid]):
            break
    li = i + 1 if i > 0 else 1

    # Find right limit (ri)
    for i in range(mid, llen - 1):
        if dline[i] <= 0.01 and line[i] > -0.1 * abs(line[mid]):
            break
    ri = i if i < llen - 1 else llen

    # Additional left and right refinements
    for i in range(li, mid):
        if line[i] < 0:
            break
    left_bound = i if i != mid else 1

    # Initialize `j` before the loop to avoid UnboundLocalError
    j = llen
    for j in range(mid, ri):
        if line[j] > 0:
            break
    right_bound = j - 1 if j < llen else llen

    leni = [left_bound, right_bound]

    # Check conditions for length calculation
    if not leni or (leni[0] == 1 and leni[1] == llen):
        return 0
    elif leni[0] == 1:
        sline_r = abs(line[leni[1]]) / (abs(line[leni[1]]) + abs(line[leni[1] + 1]))
        rp = sline_r + leni[1] - mid
        lp = 100
    elif leni[1] == llen:
        sline_l = abs(line[leni[0]]) / (abs(line[leni[0]]) + abs(line[leni[0] - 1]))
        lp = sline_l + mid - leni[0]
        rp = 100
    else:
        sline_l = abs(line[leni[0]]) / (abs(line[leni[0]]) + abs(line[leni[0] - 1]))
        sline_r = abs(line[leni[1]]) / (abs(line[leni[1]]) + abs(line[leni[1] + 1]))
        lp = sline_l + mid - leni[0]
        rp = sline_r + leni[1] - mid

    # Determine final radius
    if lp <= 0 < rp < 100:
        rr = rp
    elif rp <= 0 < lp < 100:
        rr = lp
    elif 0 < lp < 100 and 0 < rp < 100:
        rr = (lp + rp) / 2
    elif 0 < lp < 100 and rp == 100:
        rr = lp
    elif 0 < rp < 100 and lp == 100:
        rr = rp
    else:
        rr = 0

    # Calculate total diameter
    total = rr * 2
    return total


def get_volume(sample, perct, x_space, y_space, z_space):
    """
    Calculates the volume of a region in `sample` using a Difference of Gaussian method.
    
    Parameters:
    sample (ndarray): 2D region of interest from the 3D image.
    perct (float): Threshold percentage for length calculation.
    
    Returns:
    float: Calculated volume in cubic micrometers.
    """
    rsize = 5  # Resizing factor for enlargement
    
    # Resize the sample image by the specified factor
    enlarged = resize(sample, (sample.shape[0] * rsize, sample.shape[1] * rsize), anti_aliasing=True)
    I, J = enlarged.shape

    # Define central indices in enlarged sample
    cI = (I + 1) // 2
    cJ = (J + 1) // 2

    # Extract horizontal and vertical center lines and calculate lengths
    lineI = enlarged[cI, :]
    dI = getllength(lineI, perct)
    lineJ = enlarged[:, cJ]
    dJ = getllength(lineJ, perct)

    # Adjust dimensions if any length is zero
    if dI == 0 and dJ == 0:
        return 0
    elif dI == 0:
        dI = dJ
    elif dJ == 0:
        dJ = dI

    # Check for consistency between dimensions
    if abs(dI - dJ) / min(dI, dJ) > 0.8:
        dI = 0
        dJ = 0

    # Calculate radius and volume
    r = (dI + dJ) / 4
    r /= rsize  # Scale radius back to original size
    v = (4 * np.pi * r**3) / 3  # Volume of a sphere formula

    # Convert volume using voxel dimensions (cubic micrometers)
    v *= x_space * y_space * z_space

    return v

def getminindex(x):
    """
    Finds the integer part of the element with the minimum fractional part in `x`.
    
    Parameters:
    x (ndarray): Array of values.
    
    Returns:
    int: Integer part of the element with the minimum fractional part.
    """
    indices = np.floor(x).astype(int)  # Integer parts
    fractional_parts = x - indices     # Fractional parts

    # Identify the index of the smallest fractional part
    min_index = np.argmin(fractional_parts)
    
    return indices[min_index]

def getproperty_mindx(img, m):
    """
    Retrieves indices of the minimum intensity values within each labeled region in `m`.
    
    Parameters:
    img (ndarray): 3D intensity image.
    m (ndarray): 3D mask or label matrix.
    
    Returns:
    ndarray: Indices of minimum intensity values within each labeled region.
    """
    # Find indices in m where m > 0 (non-zero labels)
    infd = np.where(m > 0)
    
    # Extract intensities at these valid indices
    Int = img[infd].flatten()
    Int[Int == 1] = 0.999  # Adjust values to avoid issues with exactly 1
    Int[Int == 0] = 0.001  # Adjust values to avoid issues with exactly 0
    
    # Get linear indices of these non-zero locations
    ii = np.ravel_multi_index(infd, img.shape)
    
    # Get the labels in m at these indices
    LL = m[infd].flatten()
    
    # Combine intensities and linear indices
    inputx = Int + ii
    
    # Use accumarray-like functionality to find the minimum index for each label
    unique_labels = np.unique(LL)
    tG = np.zeros_like(unique_labels, dtype=int)
    
    for i, label in enumerate(unique_labels):
        label_indices = np.where(LL == label)
        tG[i] = getminindex(inputx[label_indices])

    return tG

def get_all_vol(kimg, m, perct, x_space, y_space, z_space):
    """
    Calculates volumes for each region in `kimg` based on `m` using a threshold `perct`.
    
    Parameters:
    kimg (ndarray): 3D image containing intensity values.
    m (ndarray): 3D mask or label matrix where each label corresponds to a unique region.
    perct (float): Threshold percentage for volume calculation.
    
    Returns:
    ndarray: Array of volumes for each region identified in `m`.
    """
    # Get minimum intensity indices for each labeled region
    tG = getproperty_mindx(kimg, m)
    
    # Convert linear indices to 3D coordinates
    I, J, K = np.unravel_index(tG, kimg.shape)
    cends = np.column_stack((I, J, K))

    l = 5  # Sample region radius
    rs = np.zeros(len(cends))  # Array to store volumes
    aa, bb, cc = kimg.shape

    # Calculate volumes for each region center
    for i, (ci, cj, ck) in enumerate(cends):
        # Check if the sample region is within bounds
        if ci <= l or ci + l >= aa or cj <= l or cj + l >= bb or ck >= cc:
            continue
        
        # Extract a sample centered at (ci, cj, ck) and calculate volume
        samp = kimg[ci - l:ci + l + 1, cj - l:cj + l + 1, ck]
        rs[i] = get_volume(samp, 1 - perct, x_space, y_space, z_space)

    return rs

def kidney_volume(IMG, x_value, y_value, z_value):
    """
    Calculates the volume of non-zero regions in a 3D image (kidney volume).
    
    Parameters:
    IMG (ndarray): 3D binary or intensity image where kidney regions are labeled as > 0.
    
    Returns:
    tuple: (kidneynum, kv) where kidneynum is the number of non-zero voxels,
           and kv is the calculated volume.
    """
    # Find indices of non-zero elements (kidney voxels)
    kidneyindex = np.count_nonzero(IMG > 0)
    kidneynum = kidneyindex
    
    # Calculate volume using the given voxel dimensions (cubic micrometers)
    kv = kidneynum * x_value * y_value * z_value
    
    return kidneynum, kv

def save_rgb_slices_as_png(array, target_folder):
    """
    Save a 3D RGB NumPy array as PNG images in a target folder.
    
    Parameters:
        array (numpy.ndarray): A 3D array with shape [:, :, n, 3], where the last dimension is the RGB channels.
        target_folder (str): Path to the folder where the images will be saved.
    """
    # Validate the input array shape
    if len(array.shape) != 4 or array.shape[3] != 3:
        raise ValueError("Input array must have shape [:, :, n, 3] where n is the number of slices.")
    
    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)
    
    # Save each slice as a PNG
    for i in range(array.shape[2]):
        # Extract the ith slice
        slice_rgb = array[:, :, i, :]
        
        # Ensure the data is in uint8 format for saving as an image
        if slice_rgb.dtype != np.uint8:
            slice_rgb = (255 * (slice_rgb / np.max(slice_rgb))).astype(np.uint8)
        
        # Create an image object
        img = Image.fromarray(slice_rgb, 'RGB')
        
        # Save the image with the name `i.png`
        slice_filename = os.path.join(target_folder, f"{i}.png")
        img.save(slice_filename)

    print(f"All slices have been saved in {target_folder}")
