import io

import cv2
import numpy
from PIL import Image


def image_from_bytes(image_bytes):
    """
    Convert file from Flask request.data to 3D Numpy array
    :param image_file: file from Flask request.data, bytes object
    :return: 3D Numpy array, np.array()
    """
    # Check image to correct format
    image = Image.open(io.BytesIO(image_bytes))
    return image