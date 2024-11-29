import torch
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp

# NSIDE = 32

def resolution(N):
    print(hp.nside2resol(N, arcmin=True)/60)

def npix(n):
    print(hp.nside2npix(n))

if __name__ == '__main__':
    resolution(128)
    npix(128)
    print(196608 ** 0.5)
