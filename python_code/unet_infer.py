import os
import sys

from python_code.model import *
from python_code.dataGen import *

import skimage.io as io
import skimage.transform as trans
import tensorflow as tf
from keras.models import *

import warnings
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF messages, including errors
tf.get_logger().setLevel('ERROR')  # Suppress warnings and errors below ERROR level


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller executable."""
    try:
        # PyInstaller stores temp path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        # If not running as a PyInstaller bundle, use the current directory
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def infer(testfolderdir, outputfolderdir, sectornum):
    reconstructed_model = load_model(resource_path("kidney_427.hdf5")) # kidney_427 or kidney_blobgan

    list = os.listdir(testfolderdir)
    num_image = len(list)

    testGene = testGenerator(testfolderdir,num_image=num_image)
    results = reconstructed_model.predict_generator(testGene,num_image)
    saveResult(outputfolderdir,sectornum, results)
