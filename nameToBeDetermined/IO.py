import os
import torch
import mrcfile
import tifffile


def cryoRead(inputFile: str, tensor: torch.Tensor) -> torch.Tensor:
    suffix = os.path.splitext(inputFile)
    if suffix == ".mrc" or suffix == '.mrcs' or suffix == '.map':
        with mrcfile.open(inputFile, mode='r') as mrc:
            tensor = torch.from_numpy(mrc.data)
    elif suffix == ".tif" or suffix == ".tiff":
        tensor = torch.from_numpy(tifffile.imread(inputFile))
    else:
        print("Error: cryoRead: unsupported file format")
    return tensor


def cryoWrite(inputTensor: torch.Tensor, outputFile: str):
    suffix = os.path.splitext(outputFile)
    if suffix == ".mrc" or suffix == '.mrcs' or suffix == '.map':
        with mrcfile.new(outputFile, overwrite=True) as mrc:
            mrc.set_data(inputTensor.numpy())
    elif suffix == '.tiff' or suffix == '.tif':
        tifffile.imwrite(outputFile,
                         inputTensor.numpy(),
                         photometric='minisblack')
