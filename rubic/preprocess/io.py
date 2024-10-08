import mrcfile
import tifffile
import numpy as np
import os


def read_img(img_path: str) -> np.ndarray:
    file_extension = os.path.splitext(img_path)[1]
    if file_extension == '.tiff' or file_extension == '.tif':
        img_array = tifffile.imread(img_path)
    elif file_extension == '.mrc' or file_extension == '.mrcs':
        with mrcfile.open(img_path) as mrc:
            img_array = np.array(mrc.data)
    else:
        raise Exception(
            "Image format not supported; Only tiff and MRC format are supported"
        )
    return img_array


def write_img(img_array: np.ndarray, img_path: str):
    file_extension = os.path.splitext(img_path)[1]
    if file_extension == '.tiff' or file_extension == '.tif':
        tifffile.imwrite(img_path, img_array, photometric='minisblack')
    elif file_extension == '.mrc' or file_extension == '.mrcs':
        with mrcfile.new(img_path) as mrc:
            mrc.set_data(img_array)
    else:
        raise Exception(
            "Image format not supported; Only tiff and MRC format are supported"
        )
