#!/usr/bin/env python

import torch
import typing
import mrcfile
import os
import math as Math
import numpy as np


def ctf(box_size=300,
        pixel_size=1.0,
        defocus=10000,  # (U + V) / 2
        astigmatism=200, # (U - V) / 2
        ast_angle=90,
        Cs=2.7,
        voltage=300,
        amp_contrast=0.1,
        phase_shift=0) -> torch.Tensor:
    '''
    Generate a 2D CTF image for given parameters
    '''
    nx, ny = box_size, box_size
    wavelength = wavelength(voltage)
    pc = 


def wavelength(voltage=300):
    '''
    Return wavelength for given voltage in [angstrom]
    '''
    candidate_voltage = torch.tensor([120,200,300])
    candidate_wavelength = torch.tensor([3.3492e-2, 2.5079e-2, 1.9678e-2])
    idx = torch.where(candidate_voltage==voltage)
    return candidate_wavelength[idx]


if __name__ == '__main__':
    pass
