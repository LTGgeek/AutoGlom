import cv2
import os

def bilinear_resize(img, target_width, target_height):
    """
    Perform bilinear resizing on the input image to the target resolution.
    
    :param img: Input image (numpy array)
    :param target_width: Target width of the output image
    :param target_height: Target height of the output image
    :return: Resized image
    """
    return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

def process_image_batch( input_path, output_path, target_width, target_height):
    for imgfile in os.listdir(input_path):
        img_path = os.path.join(input_path, imgfile)
        print(f"Processing: {img_path}")
        
        # Read the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Error reading image: {img_path}")
            continue
        
        # Perform bilinear resizing to the target resolution
        resized_img = bilinear_resize(img, target_width, target_height)
        
        # Use the original filename but change the extension to .png
        output_filename = os.path.splitext(imgfile)[0] + '.png'
        output_file_path = os.path.join(output_path, output_filename)
        
        # Save the resized image
        cv2.imwrite(output_file_path, resized_img)

