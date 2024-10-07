import torch
import numpy as np
import mrcfile
from normalize import norm
import cv2
import tifffile

def tensor2mrc(input_tensor:torch.Tensor, location:str):
    np_arr = norm(input_tensor).numpy()
    with mrcfile.new(location, overwrite=True) as mrc:
        mrc.set_data(np_arr.astype(np.float32))

def tensor2tiffstack(input_tensor:torch.Tensor, location:str, dtype=np.float32, norm=norm):
    np_arr = norm(input_tensor).numpy()
    tifffile.imwrite(location, np_arr.astype(dtype))

def mrc2tensor(mrc_path, dtype=torch.float32, norm=norm):
    with mrcfile.open(mrc_path) as mrc:
        arr = np.copy(mrc.data)
        return norm(torch.from_numpy(arr).to(dtype))

def tiffstack2tensor(input_tiff:str, dtype=torch.float32, norm=norm):
    # arr = cv2.imreadmulti(input_tiff, cv2.IMREAD_GRAYSCALE)
    arr = tifffile.imread(input_tiff)
    return norm(torch.from_numpy(arr).to(dtype))

def mrc2tiffstack(mrc_path, tiffstack_path):
    with mrcfile.open(mrc_path) as mrc:
        arr = mrc.data
        tifffile.imwrite(tiffstack_path, arr)

def tensor2tif(tensor:torch.Tensor, location:str, dtype=np.float32, norm=norm):
    arr = norm(tensor).numpy()
    cv2.imwrite(location, arr.astype(dtype))
