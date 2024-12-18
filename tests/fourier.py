import torch
import mrcfile
import numpy as np
import cv2

with mrcfile.open('data/emd_7770.mrc') as mrc:
    data = mrc.data
data = np.array(data)

slice = data[155]

fft = np.fft.fftn(data)
# fft = np.fft.fftshift(fft)
cv2.imwrite('one.tif', np.abs(np.fft.fft2(slice)))
cv2.imwrite('two.tif', np.abs(fft[155]))
