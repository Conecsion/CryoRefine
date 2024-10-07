import mrcfile
import tifffile
import numpy as np
import os


def readImg(imgPath: str) -> np.ndarray:
    fileExtension = os.path.splitext(imgPath)[1]
    if fileExtension == 'tiff' or fileExtension == 'tif':
        imgArray = tifffile.imread(imgPath)
    elif fileExtension == 'mrc' or fileExtension == 'mrcs':
        with mrcfile.open(imgPath) as mrc:
            imgArray = np.array(mrc.data)
    else:
        raise Exception(
            "Image format not supported; Only tiff and MRC format are supported"
        )
    return imgArray


def writeImg(imgArray: np.ndarray, imgPath: str):
    fileExtension = os.path.splitext(imgPath)[1]
    if fileExtension == 'tiff' or fileExtension == 'tif':
        tifffile.imwrite(imgPath, imgArray, photometric='minisblack')
    elif fileExtension == 'mrc' or fileExtension == 'mrcs':
        with mrcfile.new(imgPath) as mrc:
            mrc.set_data(imgArray)
    else:
        raise Exception(
            "Image format not supported; Only tiff and MRC format are supported"
        )
