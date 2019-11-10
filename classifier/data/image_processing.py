import io

from PIL import Image


def image_from_file(file):
    """
    Convert file to Pillow Image
    :param file: image file
    :return: Pillow Image
    """
    image = Image.open(io.BytesIO(file))
    return image
