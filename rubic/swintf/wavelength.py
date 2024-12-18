import numpy as np


def wavelength(voltage):
    h = 6.626e-34
    e = 1.602e-19
    c = 3.000e8
    m0 = 9.109e-31
    # return h / np.sqrt(2 * m0 * e * voltage) / np.sqrt(1 + e * voltage /
    #                                                    (2 * m0 * c**2)) * 1e10
    return h / np.sqrt(2*e*voltage*m0) * np.sqrt(1-(2*e*voltage/(m0*c**2)))

if __name__ == '__main__':
    for i in (120,200,300):
        print(wavelength(i))
        print(i)
        print('\n')
