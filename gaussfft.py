import numpy as np
from numpy.fft import fft2, ifft2, fftshift

def gaussfft(pic, t):
		
    row, col = np.shape(pic)
    x, y = np.meshgrid(np.linspace(-(row-1)/2, (row-1)/2, row), np.linspace(-(col-1)/2, (col-1)/2, col))
    
    gaussian_filter = (1/(2*np.pi*t))*np.e**(-(x**2+y**2)/(2*t))
    
    Fhat = fft2(pic)
    Ghat = fft2(fftshift(gaussian_filter))
    result = ifft2(Fhat*Ghat)

    return result