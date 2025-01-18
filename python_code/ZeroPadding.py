from PIL import Image
import os
import cv2
import numpy as np

def pad_centered_image(imgfolder, padimgfolder, size, padsize, pad_color='black'):
    """
    Pad images to a target size while maintaining center alignment.
    
    Args:
        imgfolder (str): Input folder containing images
        padimgfolder (str): Output folder for padded images
        size (str or int): Original image size
        padsize (int): Target size after padding
        pad_color (str): 'black' or 'white', default 'black'
    
    Example:
        # Pad images to 256x256 with black background
        pad_centered_image('input_folder', 'output_folder', 169, 256)
        
        # Pad images to 512x512 with white background
        pad_centered_image('input_folder', 'output_folder', 256, 512, 'white')
    """
    # Validate pad_color argument
    if pad_color not in ['black', 'white']:
        raise ValueError("pad_color must be either 'black' or 'white'")
        
    # Set padding intensity value based on color choice
    padint = 0 if pad_color == 'black' else 255

    for oimgfile in os.listdir(imgfolder):
        tpath = os.path.join(imgfolder, oimgfile)
        opath = os.path.join(padimgfolder, oimgfile)

        oim = cv2.imread(tpath, 0)

        oimsize = np.shape(oim)
        oimw = oimsize[0]
        oimh = oimsize[1]

        oimleftpad = int(np.floor((padsize - oimw) / 2))
        oimuppad = int(np.floor((padsize - oimh) / 2))

        # Create padded image array
        oimnew = np.zeros((padsize, padsize))
        oimnew.fill(padint)  # Fill with chosen padding color

        # Place original image in center of padded array
        oimnew[oimleftpad:oimleftpad+int(size), oimuppad:oimuppad+int(size)] = oim

        # Convert to PIL image and save
        im = Image.fromarray(oimnew)
        im = im.convert("L")
        im.save(opath)