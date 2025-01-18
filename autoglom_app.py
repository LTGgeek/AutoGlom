# Standard Library Imports
# Add at the start of demo.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))


import shutil
import threading
import json

# Tkinter GUI Imports
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox

# External Library Imports
from PIL import Image, ImageTk

# Local Module Imports
from python_code.ZeroPadding import pad_centered_image
from python_code.kidney_segmentation import main as kidney_segmentation_main1
from python_code.upsampling import process_image_batch
from python_code.unet_infer import infer as unet_infer
from convert_nrrd_to_png import convert_nrrd_to_png
from uhdog.uhdog import run_uhdog
# lazy load for executor
def import_tf():
    global tf
    import tensorflow as tf
    return tf

# Global variables for the viewer
image_list_1, image_list_2, image_list_3 = [], [], []
current_image_index = 0
img_label_1, img_label_2, img_label_3 = None, None, None
slider = None
view_name_labels = []
script7_frame = None
sector = None  # This should be set when the initial folder is selected
original_size = None    # This should be set when the initial folder is selected
size = None  # This should be set when the initial folder is selected
sslice = None
eslice = None
results_frame = None
kidney_metadata = None
bdot_file = None
image_file_entry = None
kidney_annotation_entry = None
medulla_annotation_entry = None
blackdot_annotation_entry = None
sector_entry = None
target_folderdir_entry = None
slice_axis_var = None
setup_window = None

class ToolTip(object):
    def __init__(self, widget, text, wrap_length=250):
        self.widget = widget
        self.text = text
        self.wrap_length = wrap_length  # Maximum width in pixels before wrapping
        self.tooltip = None
        self.widget.bind('<Enter>', self.show_tooltip)
        self.widget.bind('<Leave>', self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)

        label = tk.Label(self.tooltip, text=self.text, 
                      justify=tk.LEFT,
                      background="#ffffe0", 
                      relief=tk.SOLID, 
                      borderwidth=1,
                      font=("Arial", "10", "normal"),
                      wraplength=self.wrap_length,  # Enable text wrapping
                      padx=5,  # Add horizontal padding
                      pady=3)  # Add vertical padding
        label.pack()

        # Position the tooltip considering screen edges
        screen_width = self.widget.winfo_screenwidth()
        tooltip_width = label.winfo_reqwidth()
        
        # Adjust x position if tooltip would go off screen
        if x + tooltip_width > screen_width:
            x = screen_width - tooltip_width - 20

        self.tooltip.wm_geometry(f"+{x}+{y}")

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

def create_labeled_entry(parent, label_text, default_value, tooltip_text):
    """Create a label, entry, and tooltip icon group"""
    frame = tk.Frame(parent)
    frame.pack(pady=2)
    
    label = tk.Label(frame, text=label_text)
    label.pack(side=tk.LEFT, padx=(0, 5))
    
    entry = tk.Entry(frame)
    entry.insert(0, default_value)
    entry.pack(side=tk.LEFT)
    
    help_button = tk.Label(frame, text="?", font=("Arial", 8), 
                          relief="raised", width=2, cursor="question_arrow")
    help_button.pack(side=tk.LEFT, padx=(5, 0))
    
    ToolTip(help_button, tooltip_text)
    
    return entry

# Function to browse for the folder directory
def browse_folder(entry_field):
    folderdir = filedialog.askdirectory()
    if folderdir:  # If a folder is selected
        entry_field.delete(0, tk.END)  # Clear current entry
        entry_field.insert(0, folderdir)  # Insert the selected path

def browse_file(entry_field):
    filepath = filedialog.askopenfilename(
        title='Select NRRD file',
        filetypes=[('NRRD files', '*.nrrd')]
    )
    if filepath:  # If a file is selected
        entry_field.delete(0, tk.END)  # Clear current entry
        entry_field.insert(0, filepath)  # Insert the selected path

def read_json_results(sector, target_folderdir):
    json_path = os.path.join(target_folderdir, f"{sector}_results", "results.json")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        messagebox.showerror("Error", f"Results file not found: {json_path}")
        return None
    except json.JSONDecodeError:
        messagebox.showerror("Error", f"Invalid JSON in results file: {json_path}")
        return None
    

def organize_converted_images(temp_png_folder, target_folderdir, sector, size):
    """
    Organize converted PNG images into the required directory structure.
    Returns a tuple of initial image paths for the viewer.
    """
    global kidney_metadata
    # Create base directories
    newpath0 = os.path.join(target_folderdir, f"{sector}_{size}")
    newpath = os.path.join(target_folderdir, f"{sector}_256")
    newpath2 = os.path.join(target_folderdir, f"{sector}_512")

    # Define all required directories
    directories = [
        (newpath0, f"{sector}_{size}_black"),
        (newpath0, f"{sector}_{size}_white"),
        (newpath0, f"{sector}_kidney_mask_cvhull_{size}"),
        (newpath0, f"{sector}_medulla_mask_cvhull_{size}"),
        (newpath0, f"{sector}_bdot_mask_cvhull_{size}"),
        (newpath0,f"{sector}_subracted_mask_{size}"),
        (newpath, f"{sector}_256_black"),
        (newpath, f"{sector}_256_white"),
        (newpath, f"{sector}_kidney_mask_cvhull_256"),
        (newpath, f"{sector}_medulla_mask_cvhull_256"),
        (newpath, f"{sector}_bdot_mask_cvhull_256"),
        (newpath2, f"{sector}_kidney_seg_white_mask_512"),
        (target_folderdir, f"{sector}_results"),
        (target_folderdir, f"{sector}_output"),
        (target_folderdir, os.path.join("U-Net_Data", sector))
    ]

    # Create all directories
    for base_path, sector_path in directories:
        os.makedirs(os.path.join(base_path, sector_path), exist_ok=True)

    # Copy images from temp folder to appropriate locations
    # try:
        # Copy original images
    source_masked_dir = os.path.join(temp_png_folder, 'masked_cropped', 'black_mask', 'global_standardization')
    print(source_masked_dir)
    print("files: " , len(os.listdir(source_masked_dir)))
    target_dir = os.path.join(newpath0, f"{sector}_{size}_black")
    if os.path.exists(source_masked_dir):
        for filename in sorted(os.listdir(source_masked_dir)):
            if filename.endswith('.png'):
                shutil.copy(
                    os.path.join(source_masked_dir, filename),
                    os.path.join(target_dir, filename)
                )

    source_masked_dir_white = os.path.join(temp_png_folder, 'masked_cropped', 'white_mask', 'global_standardization')
    print(source_masked_dir_white)
    print("files: " , len(os.listdir(source_masked_dir_white)))
    target_dir_white = os.path.join(newpath0, f"{sector}_{size}_white")
    if os.path.exists(source_masked_dir_white):
        for filename in sorted(os.listdir(source_masked_dir)):
            if filename.endswith('.png'):
                shutil.copy(
                    os.path.join(source_masked_dir_white, filename),
                    os.path.join(target_dir_white, filename)
                )

    # Copy annotation images
    source_annotation_dir = os.path.join(temp_png_folder, 'masked_cropped', 'annotations')
    print(source_annotation_dir)
    print("files: " , len(os.listdir(source_annotation_dir)))
    target_annotation_dir = os.path.join(newpath0, f"{sector}_kidney_mask_cvhull_{size}")
    if os.path.exists(source_annotation_dir):
        for filename in sorted(os.listdir(source_annotation_dir)):
            if filename.endswith('.png'):
                shutil.copy(
                    os.path.join(source_annotation_dir, filename),
                    os.path.join(target_annotation_dir, filename)
                )

    # Copy medulla annotation images
    print("Copying medulla annotations")
    source_annotation_dir_m = os.path.join(temp_png_folder, 'medulla_annotations', 'annotations')
    print(source_annotation_dir_m)
    print("files: " , len(os.listdir(source_annotation_dir_m)))
    target_annotation_dir_m = os.path.join(newpath0, f"{sector}_medulla_mask_cvhull_{size}")
    if os.path.exists(source_annotation_dir_m):
        print(source_annotation_dir_m)

        for filename in sorted(os.listdir(source_annotation_dir_m)):
            if filename.endswith('.png'):
                print(filename)
                shutil.copy(
                    os.path.join(source_annotation_dir_m, filename),
                    os.path.join(target_annotation_dir_m, filename)
                )

    # Copy subtracted annotation images
    print("Copying subtracted annotations")
    source_annotation_dir_m = os.path.join(temp_png_folder, 'subtracted_annotations', 'annotations')
    print(source_annotation_dir_m)
    print("files: " , len(os.listdir(source_annotation_dir_m)))
    target_annotation_dir_m = os.path.join(newpath0, f"{sector}_subracted_mask_{size}")
    if os.path.exists(source_annotation_dir_m):
        print(source_annotation_dir_m)

        for filename in sorted(os.listdir(source_annotation_dir_m)):
            if filename.endswith('.png'):
                print(filename)
                shutil.copy(
                    os.path.join(source_annotation_dir_m, filename),
                    os.path.join(target_annotation_dir_m, filename)
                )

    # Copy kidney bdot mask
    if bdot_file is not None:
        source_masked_dir_bdot = os.path.join(temp_png_folder, 'bdot_annotations', 'annotations')
        print(source_masked_dir_bdot)
        print("files: " , len(os.listdir(source_masked_dir_bdot)))
        target_dir_bdot = os.path.join(newpath0, f"{sector}_bdot_mask_cvhull_{size}")
        if os.path.exists(source_masked_dir_bdot):
            for filename in sorted(os.listdir(source_masked_dir_bdot)):
                if filename.endswith('.png'):
                    shutil.copy(
                        os.path.join(source_masked_dir_bdot, filename),
                        os.path.join(target_dir_bdot, filename)
                    )

    # Copy metadata json
    source_file = os.path.join(temp_png_folder, 'metadata.json')
    target_file = os.path.join(newpath0, 'metadata.json')
    if os.path.exists(source_file):
        shutil.copy(source_file, target_file)
    
    # open metadata json
    with open(target_file, 'r') as f:
        kidney_metadata = json.load(f)
        kidney_metadata = process_metadata(kidney_metadata)
    
    execute_zero_padding()

    return target_dir, target_annotation_dir, target_annotation_dir_m
    
        

    # except Exception as e:
    #     messagebox.showerror("Error", f"Failed to organize images: {str(e)}")
    #     return None, None


def process_metadata(metadata):
    # Determine the axis and corresponding bounding box key based on slice_axis
    axes = metadata.get("slice_axis")
    print(f"axes: {axes}")
    
    if axes == 0:
        axis_label = 'x'
        bounding_box_min_key, voxel_key = 'min_x', 'x'
    elif axes == 1:
        axis_label = 'y'
        bounding_box_min_key, voxel_key = 'min_y', 'y'
    elif axes == 2:
        axis_label = 'z'
        bounding_box_min_key, voxel_key = 'min_z', 'z'
    else:
        raise ValueError("Invalid axis value. Must be 0, 1, or 2.")
    
    # Calculate adjusted kidney range
    kidney_min = metadata["kidney_range"]["min"] - metadata["adjusted_bounding_box"][bounding_box_min_key]
    kidney_max = metadata["kidney_range"]["max"] - metadata["adjusted_bounding_box"][bounding_box_min_key]
    
    # Calculate adjusted medulla range
    medulla_min = metadata["medula_range"]["min"] - metadata["adjusted_bounding_box"][bounding_box_min_key]
    medulla_max = metadata["medula_range"]["max"] - metadata["adjusted_bounding_box"][bounding_box_min_key]
    
    # Get the voxel spacing for the active axis
    voxel_spacing = metadata["voxel_spacing"]
    voxel_spacing[voxel_key], voxel_spacing['z'] = voxel_spacing['z'], voxel_spacing[voxel_key] 
    
    return {
        "kidney_min": kidney_min,
        "kidney_max": kidney_max,
        "medulla_min": medulla_min,
        "medulla_max": medulla_max,
        "voxel_spacing": voxel_spacing
    }
        



def create_folder_structure():
    global sector, target_folderdir, bdot_file, size, image_list_1, image_list_2, image_list_3
    
    # Get the file paths from the global entry fields
    image_file = image_file_entry.get()
    kidney_annotation = kidney_annotation_entry.get()
    medulla_annotation = medulla_annotation_entry.get()
    slice_axis = slice_axis_var.get()
    blackdot_file = blackdot_annotation_entry.get()
    blackdot_file = blackdot_file if os.path.exists(blackdot_file) else None
    bdot_file = blackdot_file
    
    # Validate file selections
    if not all([image_file, kidney_annotation]):
        messagebox.showerror("Error", "Please select all required NRRD files.")
        return
    
    # Get sector name and target directory
    sector = sector_entry.get().replace(" ", "_")
    target_folderdir = target_folderdir_entry.get()

    if not all([sector, target_folderdir]):
        messagebox.showerror("Error", "Please fill in all fields.")
        return

    # Create temporary output folder for PNG conversion
    temp_png_folder = os.path.join(target_folderdir, "_temp_png")
    os.makedirs(temp_png_folder, exist_ok=True)

    if slice_axis == "X":
        slice_direction = 0 
    elif slice_axis == "Z":
        slice_direction = 2 
    elif slice_axis == "Y":  
        slice_direction = 1

    # Convert NRRD files to PNGs
    convert_nrrd_to_png(
        image_file=image_file,
        annotation_file1=kidney_annotation,
        annotation_file2=medulla_annotation,
        output_folder=temp_png_folder,
        slice_axis=slice_direction,
        sigmoid_alpha=0.1,
        threshold=None,
        scaling_method='global_standardization',
        img_name_prefix=sector,
        bdot_file=blackdot_file
    )

    # Get image size and organize files
    source_masked_dir = os.path.join(temp_png_folder, 'masked_cropped', 'black_mask', 'global_standardization')
    if os.path.exists(source_masked_dir):
        for filename in sorted(os.listdir(source_masked_dir)):
            if filename.endswith('.png'):
                size = Image.open(os.path.join(source_masked_dir, filename)).size[0]
                break
    
    # Organize the converted images and get initial paths
    image_dir, annotation_dir, medulla_annotation_dir = organize_converted_images(
        temp_png_folder, target_folderdir, sector, size)

    # Execute subsequent processing steps
    execute_zero_padding()
    execute_script_3(256)

    return image_dir, annotation_dir, medulla_annotation_dir

# Function to load images from a directory
def load_images_from_folder(folder):
    images = []
    # Sort by extracting numerical part if filenames contain numbers
    for filename in sorted(os.listdir(folder), key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = Image.open(os.path.join(folder, filename))
            if img.size[0] > 256 or img.size[1] > 256:
                img = img.resize((256, 256))
            images.append(img.copy()) 
    print(f"Loaded {len(images)} images from {folder}.")
    return images

def display_json_results(data, frame):
    # Clear any existing widgets in the frame
    for widget in frame.winfo_children():
        widget.destroy()

    # Show the frame again with the same grid parameters as original creation
    frame.grid(row=3, column=0, columnspan=3, pady=10)

    # Create a formatted string with the results
    result_text = f"""
        Results:
        Kidney Volume:        {data['kidney_volume']:.4f} mm³
        Medulla Volume:       {data['medulla_volume']:.4f} mm³
        Number of Glomeruli:  {data['n_glom']}
        Mean Glomeruli Vol.:  {data['mean_vs']:.3e} mm³
        Median Glomeruli Vol.:{data['med_vs']:.3e} mm³
        Intensity Threshold:  {data['inten_thre']:.2f}
        """
    if bdot_file:
        result_text = f"""
        Results:
            Kidney Volume:        {data['kidney_volume']:.4f} mm³
            Medulla Volume:       {data['medulla_volume']:.4f} mm³
            Black Dot Volume:     {data['blackdot_volume']:.4f} mm³
            Number of Glomeruli:  {data['n_glom']}
            Mean Glomeruli Vol.:  {data['mean_vs']:.3e} mm³
            Median Glomeruli Vol.:{data['med_vs']:.3e} mm³
            Intensity Threshold:  {data['inten_thre']:.2f}
            """

    # Create a label with the formatted text
    result_label = tk.Label(frame, text=result_text, justify=tk.LEFT, font=("Arial", 10))
    result_label.grid(row=0, column=0, pady=10)


# # Update run_script_1 function to get size from the dropdown and avoid global conflicts
# def run_script_2():
#     global sector, size, target_folderdir
    
#     run_with_loading(execute_zero_padding)

def execute_zero_padding():
    global sector, size, target_folderdir

    padding_size = 256


    # Construct paths
    sectorsizefolder = f"{sector}_{size}"
    padsizefolder = f"{sector}_{padding_size}"

    # Raw image paths
    rawimgdir = os.path.join(target_folderdir, sectorsizefolder, f"{sector}_{size}_black")
    padrawdir = os.path.join(target_folderdir, padsizefolder, f"{sector}_{padding_size}_black")

    # White image paths
    whiteimgdir = os.path.join(target_folderdir, sectorsizefolder, f"{sector}_{size}_white")
    padwhiteimgdir = os.path.join(target_folderdir, padsizefolder, f"{sector}_{padding_size}_white")

    # Edge or adaptive thresholding images paths
    mdir = os.path.join(target_folderdir, sectorsizefolder, f"{sector}_kidney_mask_cvhull_{size}")
    padmdir = os.path.join(target_folderdir, padsizefolder, f"{sector}_kidney_mask_cvhull_{padding_size}")

    # medulla mask paths
    medmaskdir = os.path.join(target_folderdir, sectorsizefolder, f"{sector}_medulla_mask_cvhull_{size}")
    padmedmaskdir = os.path.join(target_folderdir, padsizefolder, f"{sector}_medulla_mask_cvhull_{padding_size}")

    # bdot paths
    bdotdir = os.path.join(target_folderdir, sectorsizefolder, f"{sector}_bdot_mask_cvhull_{size}")
    padbdotdir = os.path.join(target_folderdir, padsizefolder, f"{sector}_bdot_mask_cvhull_{padding_size}")


    # Create directories if they don't exist
    os.makedirs(padrawdir, exist_ok=True)
    os.makedirs(padmdir, exist_ok=True)
    os.makedirs(padmedmaskdir, exist_ok=True)
    os.makedirs(padwhiteimgdir, exist_ok=True)

    pad_centered_image(rawimgdir, padrawdir, str(size), padding_size)
    pad_centered_image(mdir, padmdir, str(size), padding_size)
    pad_centered_image(medmaskdir, padmedmaskdir, str(size), padding_size)
    pad_centered_image(bdotdir, padbdotdir, str(size), padding_size)
    pad_centered_image(whiteimgdir, padwhiteimgdir, str(size), padding_size, 'white')


def execute_script_3(kidney_size):
    global sector, target_folderdir, kidney_metadata

    # Construct paths
    sectorsizefolder = f"{sector}_{kidney_size}"
    opath = os.path.join(target_folderdir, sectorsizefolder, f"{sector}_{kidney_size}_black")
    segpathw = os.path.join(target_folderdir, sectorsizefolder, f"{sector}_kidney_seg_white_mask_{kidney_size}")
    segpathb = os.path.join(target_folderdir, sectorsizefolder, f"{sector}_kidney_seg_black_mask_{kidney_size}")

    
    # Create directories if they don't exist
    for path in [segpathw, segpathb]:
        os.makedirs(path, exist_ok=True)

    # Call the main function from KidneySegmentation
    kidney_segmentation_main1(target_folderdir, sector, kidney_metadata['kidney_min'], kidney_metadata['kidney_max'], bdot=bdot_file)
    run_script_4()

    segpathb = os.path.join(target_folderdir, sectorsizefolder, f"{sector}_{kidney_size}_black")
    segpathw = os.path.join(target_folderdir, sectorsizefolder, f"{sector}_{kidney_size}_white")


def run_script_4():
    global sector, size, target_folderdir
    
    # Define input and output paths
    input_folder = f"{sector}_256_white"
    output_folder = f"{sector}_kidney_seg_white_mask_512"
    input_path = os.path.join(target_folderdir, f"{sector}_{256}", input_folder)
    output_path = os.path.join(target_folderdir, f"{sector}_{512}", output_folder)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Call the process_image_batch function
    process_image_batch(input_path, output_path, 512, 512)

    input_folder2 = f"{sector}_kidney_seg_white_mask_512"
    input_path2 = os.path.join(target_folderdir, f"{sector}_512", input_folder2)
    output_path2 = os.path.join(target_folderdir, "U-Net_Data", "Test", sector)
    
    # Check if the folder exists
    if os.path.exists(output_path2):
        # Remove the folder and all its contents
        shutil.rmtree(output_path2)

    # Create the folder
    os.makedirs(output_path2, exist_ok=True)     

    # Call the generate_kidney_data function
    files = os.listdir(input_path2)
    files.sort()

    for i, f in enumerate(files):
        src = os.path.join(input_path2, f)
        dst = os.path.join(output_path2, f"{i}.png")
        shutil.copy(src, dst)

    

def run_script_6():
    global sector, target_folderdir, script7_frame, results_frame
    
    # Hide other frames
    script7_frame.pack_forget()
    results_frame.grid_remove()
    
    def process():
        # Get progress manager
        progress_mgr = ProgressManager.get_instance()
        
        try:
            progress_mgr.update_message("Processing U-Net inference...")
            
            # Define paths
            test_folder = os.path.join(target_folderdir, "U-Net_Data", "Test", sector)
            output_folder = os.path.join(target_folderdir, f"{sector}_output")
            
            # Create output directory
            os.makedirs(output_folder, exist_ok=True)
            
            # Run inference
            unet_infer(test_folder, output_folder, sector)
            
            # Update images
            image_list_1 = load_images_from_folder(test_folder)
            image_list_2 = load_images_from_folder(output_folder)
            
            # Schedule UI update on main thread
            if hasattr(create_viewer_window, 'window'):
                create_viewer_window.window.after(0, lambda: update_images(
                    new_image_list_1=image_list_1,
                    new_image_list_2=image_list_2,
                    new_image_list_3=[],
                    view1_name="Input Images",
                    view2_name="U-Net Output",
                    view3_name=""
                ))
                
        except Exception as e:
            raise Exception(f"U-Net inference failed: {str(e)}")
    
    run_with_progress(process)

def run_script_7():
    global script7_frame, results_frame, kidney_metadata
    
    # Hide other script frames if they're visible
    for frame in [results_frame]:
        frame.grid_remove() 

    # Clear existing widgets
    for widget in script7_frame.winfo_children():
        widget.destroy()

    # Algorithm selection dropdown
    tk.Label(script7_frame, text="Select Algorithm:").pack()
    algorithm_var = tk.StringVar(script7_frame)
    algorithm_var.set("UHDoG")  # default value
    algorithm_dropdown = ttk.Combobox(script7_frame, textvariable=algorithm_var)
    algorithm_dropdown['values'] = ('UHDoG')
    algorithm_dropdown.pack()

    # Disable dummy options
    def disable_dummy_options(event):
        if algorithm_dropdown.get() != 'UHDoG':
            algorithm_dropdown.set('UHDoG')
    algorithm_dropdown.bind('<<ComboboxSelected>>', disable_dummy_options)

    # Input fields with tooltip icons
    dist_boundary_entry = create_labeled_entry(
        script7_frame,
        "Distance Boundary:",
        "2.5",
        "The gap between the kidney's outer boundary and the boundary of the preserved region. The area within this gap is removed to minimize false positives generated by the deep learning method. A larger Distance Boundary value results in a larger removal area. (default: 2.5)"
    )

    inten_thre_entry = create_labeled_entry(
        script7_frame,
        "Intensity Threshold:",
        "0.10",
        "The threshold of intensity for DoG method to smooth. A higher Intensity Threshold results in a lower number of gloms being segmented. (0-1, default: 0.10)"
    )

    unet_mask_threshold_entry = create_labeled_entry(
        script7_frame,
        "unet mask threshold:",
        "0.5",
         "The probability threshold to binarize the mask generated by UNet. A higher UNet Mask Threshold results in a lower number of gloms being segmented. (0-1, default: 0.5)"
    )

    # Execute button
    execute_button = tk.Button(script7_frame, text="Execute Analysis", 
        command=lambda: execute_analysis(
            algorithm_var.get(), 
            dist_boundary_entry.get(),
            inten_thre_entry.get(), 
            unet_mask_threshold_entry.get(),
        ))
    execute_button.pack(pady=10)
    ToolTip(execute_button, "Run glomeruli detection and counting analysis")

    script7_frame.pack()

def execute_analysis(algorithm, dist_boundary, inten_thre, unet_mask_threshold):
    """
    Execute analysis with improved progress tracking and error handling
    """
    global sector, target_folderdir, results_frame, kidney_metadata

    # Get progress manager instance
    progress_mgr = ProgressManager.get_instance()

    def validate_inputs():
        """Validate and convert input parameters"""
        try:
            return {
                'med_start': int(kidney_metadata["medulla_min"]),
                'med_end': int(kidney_metadata["medulla_max"]),
                'dist_boundary': float(dist_boundary),
                'inten_thre': float(inten_thre),
                'unet_mask_threshold': float(unet_mask_threshold),
                'perct': float(0.2),
                'x_spacing': float(kidney_metadata['voxel_spacing']['x']),
                'y_spacing': float(kidney_metadata['voxel_spacing']['y']),
                'z_spacing': float(kidney_metadata['voxel_spacing']['z']),
                'use_blackdot_mask': bool(bdot_file)
            }
        except ValueError as e:
            raise ValueError(f"Invalid input parameter: {str(e)}")
        except KeyError as e:
            raise KeyError(f"Missing required metadata: {str(e)}")

    def process():
        try:
            # Validate inputs first
            params = validate_inputs()
            
            # Update progress message
            progress_mgr.update_message("Validating parameters...")

            # Set up configuration parameters
            config_params = {
                'folder_path': target_folderdir,
                'kidney': sector,
                'sslice': kidney_metadata['kidney_min'],
                'eslice': kidney_metadata['kidney_max'],
                'med_start_slice': params['med_start'],
                'med_end_slice': params['med_end'],
                'dist_boundary': params['dist_boundary'],
                'inten_thre': params['inten_thre'],
                'perct': params['perct'],
                'unet_mask_threshold': params['unet_mask_threshold'],
                'x_space': params['x_spacing'],
                'y_space': params['y_spacing'],
                'z_space': params['z_spacing'],
                'use_blackdot_mask': params['use_blackdot_mask']
            }

            if algorithm == "UHDoG":
                # Update progress message
                progress_mgr.update_message("Preparing output directory...")

                # Set up output directory
                output_path = os.path.join(target_folderdir, f"{sector}_results")
                if os.path.exists(output_path):
                    shutil.rmtree(output_path)
                os.makedirs(output_path, exist_ok=True)

                # Update progress
                progress_mgr.update_message("Running UHDoG analysis...")

                # Run analysis
                run_uhdog(config_params)

                # Update progress
                progress_mgr.update_message("Processing results...")

                # Read and process results
                json_data = read_json_results(sector, target_folderdir)
                if not json_data:
                    raise Exception("Failed to read analysis results")

                # Schedule UI updates on main thread
                if hasattr(create_viewer_window, 'window'):
                    create_viewer_window.window.after(0, lambda: display_json_results(json_data, results_frame))
                    
                    # Load and update images
                    progress_mgr.update_message("Loading result images...")
                    image_list_1 = load_images_from_folder(
                        os.path.join(target_folderdir, f"{sector}_256", f"{sector}_256_black")
                    )
                    image_list_2 = load_images_from_folder(
                        os.path.join(target_folderdir, f"{sector}_results")
                    )
                    
                    # Update viewer
                    create_viewer_window.window.after(0, lambda: update_images(
                        new_image_list_1=image_list_1,
                        new_image_list_2=image_list_2,
                        new_image_list_3=[],
                        view1_name="Input Images",
                        view2_name="UHDoG Output",
                        view3_name=""
                    ))

            elif algorithm in ["Dummy1", "Dummy2", "Dummy3"]:
                messagebox.showinfo("Script 7", f"{algorithm} is not implemented yet.")
            else:
                raise ValueError(f"Invalid algorithm selected: {algorithm}")

        except Exception as e:
            # Log the error for debugging
            print(f"Error in execute_analysis: {str(e)}")
            # Re-raise to be caught by SafeRunner
            raise

    # Run the analysis with progress tracking
    run_with_progress(process)

# Progress Bar Manager Class
class ProgressManager:
    _instance = None
    
    def __init__(self):
        self.progress_frame = None
        self.progress_bar = None
        self.progress_label = None
        self.is_initialized = False
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ProgressManager()
        return cls._instance
    
    def initialize(self, parent_frame):
        """Initialize progress bar components"""
        if not self.is_initialized:
            # Create progress frame
            self.progress_frame = tk.Frame(parent_frame)
            self.progress_frame.grid(row=4, column=0, columnspan=3, pady=10)
            self.progress_frame.grid_remove()  # Hide initially
            
            # Create progress label
            self.progress_label = tk.Label(
                self.progress_frame, 
                text="Processing...", 
                font=("Arial", 10)
            )
            self.progress_label.pack(pady=(0, 5))
            
            # Create progress bar
            self.progress_bar = ttk.Progressbar(
                self.progress_frame, 
                mode='indeterminate', 
                length=400
            )
            self.progress_bar.pack(pady=(0, 5))
            
            self.is_initialized = True
    
    def show(self, message=None):
        """Show progress bar with optional message"""
        if not self.is_initialized:
            return
            
        if message and self.progress_label:
            self.progress_label.config(text=message)
            
        self.progress_frame.grid()
        self.progress_bar.start()
        
        # Force update the UI
        if hasattr(create_viewer_window, 'window'):
            create_viewer_window.window.update_idletasks()
    
    def hide(self):
        """Hide progress bar"""
        if not self.is_initialized:
            return
            
        self.progress_bar.stop()
        self.progress_frame.grid_remove()
        
        # Force update the UI
        if hasattr(create_viewer_window, 'window'):
            create_viewer_window.window.update_idletasks()
    
    def update_message(self, message):
        """Update progress message"""
        if self.is_initialized and self.progress_label:
            self.progress_label.config(text=message)
            if hasattr(create_viewer_window, 'window'):
                create_viewer_window.window.update_idletasks()

# Thread-safe runner
class SafeRunner:
    @staticmethod
    def run_with_progress(func, *args, **kwargs):
        """Run a function with progress bar in a thread-safe manner"""
        progress_mgr = ProgressManager.get_instance()
        
        def on_complete():
            progress_mgr.hide()
        
        def on_error(error):
            progress_mgr.hide()
            messagebox.showerror("Error", str(error))
        
        def thread_target():
            try:
                result = func(*args, **kwargs)
                # Schedule completion on main thread
                if hasattr(create_viewer_window, 'window'):
                    create_viewer_window.window.after(0, on_complete)
                return result
            except Exception as e:
                # Schedule error handling on main thread
                if hasattr(create_viewer_window, 'window'):
                    create_viewer_window.window.after(0, lambda: on_error(e))
                raise
        
        progress_mgr.show()
        
        # Start worker thread
        thread = threading.Thread(target=thread_target)
        thread.daemon = True
        thread.start()
        
        return thread

# Modified function execution helpers
def run_with_progress(func, *args, **kwargs):
    """Enhanced progress tracking for long-running operations"""
    return SafeRunner.run_with_progress(func, *args, **kwargs)

# Function to create the viewer window
def create_viewer_window():
    global img_label_1, img_label_2, img_label_3, slider
    global view_name_labels, results_frame, script7_frame, main_progress_frame, main_progress_bar 
    
    if not hasattr(create_viewer_window, 'window'):
        viewer_window = tk.Toplevel()
        create_viewer_window.window = viewer_window
        viewer_window.title("Synchronized MRI Viewer")
        viewer_window.geometry("1050x700")
        
        def go_back():
            viewer_window.withdraw()  # Hide viewer window
            if hasattr(create_setup_window, 'reset_state'):
                create_setup_window.reset_state()
            create_setup_window.window.deiconify()

        # Left Frame (For Buttons and Script Inputs)
        left_frame = tk.Frame(viewer_window, width=250, bg="lightgrey")
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Add back button at the top of left frame
        back_button = tk.Button(left_frame, text="← Back", command=go_back)
        back_button.pack(pady=(10, 20))

        # Add buttons for each script
        buttons = [
            ("Run Deep Learning Inference", lambda: run_script_6()),
            ("Glomeruli Analysis", lambda: run_script_7())
        ]

        for text, command in buttons:
            tk.Button(left_frame, text=text, command=command).pack(pady=10)

        # Right Frame (For Image Displays)
        right_frame = tk.Frame(viewer_window)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Initialize view_name_labels list before using it
        view_name_labels = []
        for i in range(3):
            label = tk.Label(right_frame, text=f"View {i+1}", font=("Arial", 12, "bold"))
            label.grid(row=0, column=i, pady=5)
            view_name_labels.append(label)

        # Store view_name_labels as an attribute of create_viewer_window
        create_viewer_window.view_name_labels = view_name_labels

        # Image Panels
        img_label_1 = tk.Label(right_frame, bg="black", width=256, height=256)
        img_label_1.grid(row=1, column=0, padx=10, pady=10)

        img_label_2 = tk.Label(right_frame, bg="black", width=256, height=256)
        img_label_2.grid(row=1, column=1, padx=10, pady=10)

        img_label_3 = tk.Label(right_frame, bg="black", width=256, height=256)
        img_label_3.grid(row=1, column=2, padx=10, pady=10)

        # Navigation Frame
        nav_frame = tk.Frame(right_frame)
        nav_frame.grid(row=2, column=0, columnspan=3, pady=10)

        prev_button = tk.Button(nav_frame, text="< Previous", command=previous_image)
        prev_button.pack(side=tk.LEFT, padx=10)

        slider = tk.Scale(nav_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=on_slider_move, length=600)
        slider.pack(side=tk.LEFT, padx=10)

        next_button = tk.Button(nav_frame, text="Next >", command=next_image)
        next_button.pack(side=tk.LEFT, padx=10)

        # Results frame
        results_frame = tk.Frame(right_frame, bg="lightgrey")
        results_frame.grid(row=3, column=0, columnspan=3, pady=10)

        # Initialize progress manager
        progress_mgr = ProgressManager.get_instance()
        progress_mgr.initialize(right_frame)  # Initialize with right frame as parent

        # Script 7 frame
        script7_frame = tk.Frame(left_frame, bg="lightgrey")
        script7_frame.pack(pady=10)
        script7_frame.pack_forget()

        # Protocol handler for window close button (X)
        def on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
                create_setup_window.window.destroy()
                viewer_window.destroy()
            
        viewer_window.protocol("WM_DELETE_WINDOW", on_closing)
        create_setup_window.window.protocol("WM_DELETE_WINDOW", on_closing)

    else:
        # If window exists but is hidden, show it.
        create_viewer_window.window.deiconify()
        for frame in [script7_frame]:
            frame.pack_forget()
        results_frame.grid_remove() 

        # Clear existing widgets
        for widget in script7_frame.winfo_children():
            widget.destroy()

    return create_viewer_window.window


def update_images(new_image_list_1=[], new_image_list_2=[], new_image_list_3=[], 
                 view1_name="View 1", view2_name="View 2", view3_name="View 3"):
    global image_list_1, image_list_2, image_list_3, current_image_index, slider

    # Get the view_name_labels from create_viewer_window
    if not hasattr(create_viewer_window, 'view_name_labels'):
        return  # Exit if window hasn't been created yet

    view_name_labels = create_viewer_window.view_name_labels

    # Update view names
    view_name_labels[0].config(text=view1_name)
    view_name_labels[1].config(text=view2_name)
    view_name_labels[2].config(text=view3_name)

    max_length = max(len(new_image_list_1), len(new_image_list_2), len(new_image_list_3))
    blank_image = Image.new('RGB', (256, 256), color='black')

    if max_length != 0:
        for i, image_list in enumerate([new_image_list_1, new_image_list_2, new_image_list_3]):
            if image_list is not None and len(image_list) > 0 and len(image_list) < max_length:
                if i == 0:
                    new_image_list_1 = [blank_image] * sslice + image_list
                elif i == 1:
                    new_image_list_2 = [blank_image] * sslice + image_list
                elif i == 2:
                    new_image_list_3 = [blank_image] * sslice + image_list
            elif image_list is None or len(image_list) == 0:
                if i == 0:
                    new_image_list_1 = [blank_image] * max_length 
                elif i == 1:
                    new_image_list_2 = [blank_image] * max_length 
                elif i == 2:
                    new_image_list_3 = [blank_image] * max_length 

    image_list_1, image_list_2, image_list_3 = new_image_list_1, new_image_list_2, new_image_list_3

    # Update slider range
    slider.config(to=max(max_length - 1, 0))

    # Ensure current_image_index is within bounds
    current_image_index = min(current_image_index, max_length - 1)
    slider.set(current_image_index)

    # Update displayed images
    display_current_images()

def display_current_images():
    global current_image_index, img_label_1, img_label_2, img_label_3

    def create_blank_image():
        return Image.new('RGB', (256, 256), color='black')

    image_lists = [image_list_1, image_list_2, image_list_3]
    img_labels = [img_label_1, img_label_2, img_label_3]

    for image_list, img_label in zip(image_lists, img_labels):
        if image_list and current_image_index < len(image_list):
            img = image_list[current_image_index]
        else:
            img = create_blank_image()
        
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk

def on_slider_move(val):
    global current_image_index
    current_image_index = int(val)
    display_current_images()

def previous_image():
    global current_image_index
    if current_image_index > 0:
        current_image_index -= 1
        slider.set(current_image_index)
        display_current_images()

def next_image():
    global current_image_index
    max_index = max(len(image_list_1), len(image_list_2), len(image_list_3)) - 1
    if current_image_index < max_index:
        current_image_index += 1
        slider.set(current_image_index)
        display_current_images()

# # Example of how another function might update the images
# def update_images_from_script(new_image_list_1, new_image_list_2, new_image_list_3):
#     update_images(new_image_list_1, new_image_list_2, new_image_list_3, 
#                   "Updated View 1", "Updated View 2", "Updated View 3")

def create_setup_window():
    global setup_window, image_file_entry, kidney_annotation_entry, medulla_annotation_entry
    global blackdot_annotation_entry, sector_entry, target_folderdir_entry, slice_axis_var
    
    if not hasattr(create_setup_window, 'window'):
        create_setup_window.window = tk.Tk()
        setup_window = create_setup_window.window
        setup_window.title("Autoglom Startup")
        setup_window.geometry("600x700")

        # Create a main frame to hold everything
        main_frame = tk.Frame(setup_window)
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)

        # Progress frame (initially hidden)
        progress_frame = tk.Frame(main_frame)
        progress_frame.pack(pady=10, fill='x', expand=True)
        progress_frame.pack_forget()

        progress_label = tk.Label(progress_frame, text="Processing...", font=("Arial", 10))
        progress_label.pack(pady=(0, 5))
        
        progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate', length=400)

        # Labels and Entry fields
        tk.Label(main_frame, text="Sector:").pack(pady=5)
        sector_entry = tk.Entry(main_frame)
        sector_entry.pack(pady=5)

        # Image NRRD file
        tk.Label(main_frame, text="Image File (NRRD):").pack(pady=5)
        image_file_frame = tk.Frame(main_frame)
        image_file_frame.pack(pady=5)
        image_file_entry = tk.Entry(image_file_frame, width=40)
        image_file_entry.pack(side=tk.LEFT)
        browse_image_button = tk.Button(image_file_frame, text="Browse", 
                                    command=lambda: browse_file(image_file_entry))
        browse_image_button.pack(side=tk.LEFT, padx=5)

        # Kidney Annotation NRRD file
        tk.Label(main_frame, text="Kidney Annotation File (NRRD):").pack(pady=5)
        kidney_annotation_frame = tk.Frame(main_frame)
        kidney_annotation_frame.pack(pady=5)
        kidney_annotation_entry = tk.Entry(kidney_annotation_frame, width=40)
        kidney_annotation_entry.pack(side=tk.LEFT)
        browse_kidney_button = tk.Button(kidney_annotation_frame, text="Browse", 
                                    command=lambda: browse_file(kidney_annotation_entry))
        browse_kidney_button.pack(side=tk.LEFT, padx=5)

        # Medulla Annotation NRRD file
        tk.Label(main_frame, text="Medulla Annotation File (NRRD):").pack(pady=5)
        medulla_annotation_frame = tk.Frame(main_frame)
        medulla_annotation_frame.pack(pady=5)
        medulla_annotation_entry = tk.Entry(medulla_annotation_frame, width=40)
        medulla_annotation_entry.pack(side=tk.LEFT)
        browse_medulla_button = tk.Button(medulla_annotation_frame, text="Browse", 
                                        command=lambda: browse_file(medulla_annotation_entry))
        browse_medulla_button.pack(side=tk.LEFT, padx=5)

        # Blackdot Annotation NRRD file
        tk.Label(main_frame, text="Blackdot Annotation File (NRRD):").pack(pady=5)
        blackdot_annotation_frame = tk.Frame(main_frame)
        blackdot_annotation_frame.pack(pady=5)
        blackdot_annotation_entry = tk.Entry(blackdot_annotation_frame, width=40)
        blackdot_annotation_entry.pack(side=tk.LEFT)
        browse_blackdot_button = tk.Button(blackdot_annotation_frame, text="Browse", 
                                        command=lambda: browse_file(blackdot_annotation_entry))
        browse_blackdot_button.pack(side=tk.LEFT, padx=5)

        # Algorithm selection dropdown
        tk.Label(main_frame, text="Scan Direction:").pack(pady=5)
        slice_axis_var = tk.StringVar(main_frame)
        slice_axis_var.set("Z")  # default value
        slice_axis_dropdown = ttk.Combobox(main_frame, textvariable=slice_axis_var, state="readonly")
        slice_axis_dropdown['values'] = ("X", "Y", "Z")
        slice_axis_dropdown.pack(pady=5)

        # Target folder directory
        tk.Label(main_frame, text="Target Folder Directory:").pack(pady=5)
        target_folder_frame = tk.Frame(main_frame)
        target_folder_frame.pack(pady=5)
        target_folderdir_entry = tk.Entry(target_folder_frame, width=40)
        target_folderdir_entry.pack(side=tk.LEFT)
        browse_target_button = tk.Button(target_folder_frame, text="Browse", 
                                    command=lambda: browse_folder(target_folderdir_entry))
        browse_target_button.pack(side=tk.LEFT, padx=5)

        # Store all buttons in a list for easy access
        all_buttons = [
            browse_image_button,
            browse_kidney_button,
            browse_medulla_button,
            browse_blackdot_button,
            browse_target_button
        ]

        # Store all entries in a list
        all_entries = [
            sector_entry,
            image_file_entry,
            kidney_annotation_entry,
            medulla_annotation_entry,
            blackdot_annotation_entry,
            target_folderdir_entry
        ]
        def reset_setup_window_state():
            """Reset the setup window to its initial state"""
            if hasattr(create_setup_window, 'progress_frame'):
                create_setup_window.progress_frame.pack_forget()
            # Reset all widget states
            for button in all_buttons:
                button.config(state='normal')
            for entry in all_entries:
                entry.config(state='normal')
            slice_axis_dropdown.config(state='readonly')
            create_button.config(state='normal')
            setup_window.update_idletasks()

        # Store these as attributes of create_setup_window
        create_setup_window.reset_state = reset_setup_window_state
        create_setup_window.progress_frame = progress_frame
        
        def set_widgets_state(state):
            # Disable/enable all buttons
            for button in all_buttons:
                button.config(state=state)
            
            # Disable/enable all entries
            for entry in all_entries:
                entry.config(state=state)
            
            # Disable/enable dropdown
            slice_axis_dropdown.config(state='readonly' if state == 'normal' else 'disabled')
            
            # Disable/enable create button
            create_button.config(state=state)
            
            # Update the window
            setup_window.update_idletasks()

        def update_status(message):
            # status_label.config(text=message)
            setup_window.update_idletasks()

        def transition_to_viewer():
            setup_window.withdraw()  # Hide setup window
            if hasattr(create_viewer_window, 'window'):
                create_viewer_window.window.deiconify()  # Show existing viewer window
            else:
                create_viewer_window()  # Create new viewer window if it doesn't exist

        def execute_with_progress():
            set_widgets_state('disabled')
            progress_frame.pack()
            progress_bar.pack(pady=5)
            progress_bar.start(10)

            def process_thread():
                global image_list_1, image_list_2, image_list_3
                try:
                    update_status("Converting NRRD files to PNG...")
                    image_dir, annotation_dir, medulla_annotation_dir = create_folder_structure()
                    
                    if image_dir and annotation_dir:
                        update_status("Loading images...")
                        image_list_1 = load_images_from_folder(image_dir)
                        image_list_2 = load_images_from_folder(annotation_dir)
                        image_list_3 = load_images_from_folder(medulla_annotation_dir)
                        
                        def show_viewer():
                            setup_window.withdraw()  # Hide setup window
                            viewer_window = create_viewer_window()
                            # Ensure proper image loading with correct labels
                            update_images(
                                new_image_list_1=image_list_1,
                                new_image_list_2=image_list_2,
                                new_image_list_3=image_list_3,
                                view1_name="Cropped Original",
                                view2_name="Kidney Segment",
                                view3_name="Cortex Segment"
                            )
                        
                        setup_window.after(0, show_viewer)
                    else:
                        raise Exception("Failed to create necessary directories")
                    
                except Exception as e:
                    error_msg = str(e)  # Capture the error message outside the lambda
                    setup_window.after(0, lambda: [
                        progress_bar.stop(),
                        progress_frame.pack_forget(),
                        set_widgets_state('normal'),
                        messagebox.showerror("Error", f"An error occurred: {str(error_msg)}")
                    ])

            thread = threading.Thread(target=process_thread)
            thread.daemon = True
            thread.start()

        # Create Directory Structure button
        create_button = tk.Button(setup_window, text="Create Directory Structure", 
                                command=execute_with_progress)
        create_button.pack(pady=20)

    else:
        # If window exists but is hidden, show it
        create_setup_window.window.deiconify()

    return create_setup_window.window

def create_folder_structure_and_continue(setup_window):
    """Wrapper function to handle the transition between windows"""
    create_folder_structure()
    setup_window.destroy()

def main():
    app = create_setup_window()
    # Set up closing protocol for main window
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            app.destroy()
    app.protocol("WM_DELETE_WINDOW", on_closing)
    app.mainloop()

if __name__ == "__main__":
    main()