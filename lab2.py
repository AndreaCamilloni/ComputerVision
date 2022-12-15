import numpy as np
#from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d, correlate2d
import matplotlib.pyplot as plt

from Functions import *
from gaussfft import gaussfft

# Either write your code in a file like this or use a Jupyter notebook.
#
# A good idea is to use switches, so that you can turn things on and off
# depending on what you are working on. It should be fairly easy for a TA
# to go through all parts of your code though.

# Exercise 1
if 0:
	print("That was a stupid idea")
        
# Partial derivatives estimated by difference operators        
def deltax():
        dxmask = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
        return dxmask

def deltay():
        return deltax().T

def Lv(inpic, shape = 'same'):
        Lx = convolve2d(inpic, deltax(), shape)
        Ly = convolve2d(inpic, deltay(), shape)
        return np.sqrt(Lx**2 + Ly**2)


def Lvvtilde(inpic, shape = 'same'):
        dxmask = np.array([[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, -1/2, 0, 1/2, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]]) # shape is 5x5 because of the convolutin type 'same'
        dymask = dxmask.T
        dxxmask = np.array([[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 1, -2, 1, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]]) 
        dyymask = dxxmask.T
        dxymask = convolve2d(dxmask, dymask, shape)

        Lx = convolve2d(inpic, dxmask, shape)
        Ly = convolve2d(inpic, dymask, shape)
        Lxx = convolve2d(inpic, dxxmask, shape)
        Lyy = convolve2d(inpic, dyymask, shape)
        Lxy = convolve2d(inpic, dxymask, shape)

        return (Lx**2*Lxx + Ly**2*Lyy + 2*Lx*Ly*Lxy) #/Lx**2 + Ly**2

def Lvvvtilde(inpic, shape = 'same'):
        dxmask = np.array([[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 1/2, 0, -1/2, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]]) # shape is 5x5 because of the convolutin type 'same'
        dymask = dxmask.T
        dxxmask = np.array([[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 1, -2, 1, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]]) 
        dyymask = dxxmask.T
        dxymask = convolve2d(dxmask, dymask, shape)
        
        dxxxmask = convolve2d(dxmask, dxxmask, "same")
        dxxymask = convolve2d(dxxmask, dymask, "same")
        dxyymask = convolve2d(dxymask, dymask, "same")
        dyyymask = convolve2d(dyymask, dymask, "same")
        
        Lx = convolve2d(inpic, dxmask, shape)
        Ly = convolve2d(inpic, dymask, shape)
        Lxxx = convolve2d(inpic, dxxxmask, shape)
        Lxxy = convolve2d(inpic, dxxymask, shape)
        Lxyy = convolve2d(inpic, dxyymask, shape)
        Lyyy = convolve2d(inpic, dyyymask, shape)
        
        return Lx * Lx * Lx * Lxxx + 3 * Lx * Lx * Ly * Lxxy + 3 * Lx * Ly * Ly * Lxyy + Ly * Ly * Ly * Lyyy
        
        
def extractedge(inpic, scale, threshold, shape = 'same'):
    gaussianSmooth = discgaussfft(inpic, scale)
    gradmagn = Lv(gaussianSmooth, "same")

    Lvv = Lvvtilde(gaussianSmooth, shape)
    Lvvv = Lvvvtilde(gaussianSmooth, shape)

    Lvmask = gradmagn > threshold
    LvvvMask = Lvvv < 0
    curves = zerocrosscurves(Lvv, LvvvMask) #Extraction of zero-crossing curves
    contours = thresholdcurves(curves, Lvmask)
    return contours
        

def houghline(pic, curves, magnitude, nrho, ntheta, threshold, nlines=20, verbose=False, scale = 0):
    acc = np.zeros((nrho, ntheta))
    x, y = magnitude.shape
    r = np.sqrt(x * x + y * y)
    rho = np.linspace(-r, r, nrho)
    theta = np.linspace(-np.pi/2, np.pi/2, ntheta)
    for i in range(len(curves[0])):
        x = curves[0][i]
        y = curves[1][i]
        curveMagn = magnitude[x][y]
        if curveMagn > threshold:
            for j in range(ntheta):
                rhoVal = x * np.cos(theta[j]) + y * np.sin(theta[j]) # compute rho, distance from origin
                rhoIndex = np.argmin(abs(rho - rhoVal)) # find the index of the closest rho value
                acc[rhoIndex][j] += 1 # increment the accumulator
                #acc[rhoIndex][j] += np.log(curveMagn) # question 10

    linepar = []
    pos, value, _ = locmax8(acc) # find local maxima
    indexvector = np.argsort(value)[-nlines:] # sort the local maxima
    pos = pos[indexvector] # sort the local maxima

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    showgrey(pic, False)
    plt.title("original")
    plt.subplot(1, 4, 2)
    showgrey(pic, False)
    for idx in range(nlines):
        thetaidxacc = pos[idx][0] 
        rhoidxacc = pos[idx][1] 
        rhoMax = rho[rhoidxacc]
        thetaMax = theta[thetaidxacc]
        linepar.append([rhoMax, thetaMax]) 

        x0 = rhoMax * np.cos(thetaMax)
        y0 = rhoMax * np.sin(thetaMax)
        dx = r * (-np.sin(thetaMax))
        dy = r * (np.cos(thetaMax))
        plt.plot([y0 - dy, y0, y0 + dy], [x0 - dx, x0, x0 + dx], "r-")


    plt.title("curves")
    plt.subplot(1, 4, 3)
    showgrey(acc, False)

    plt.title("acc")
    plt.subplot(1, 4, 4)
    overlaycurves(pic, curves)
    plt.title("sc=" + str(scale) + " t=" + str(threshold) + " nrho=" + str(nrho) + " ntheta=" +
                      str(ntheta) + " nlines=" + str(nlines))
    plt.show()
    return linepar, acc


def houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines=20, verbose=False):
    curves = extractedge(pic, scale, gradmagnthreshold, "same")
    gaussianSmooth = discgaussfft(pic, scale)
    gradmagn = Lv(gaussianSmooth, "same")

    linepar, acc = houghline(pic, curves, gradmagn, nrho, ntheta, gradmagnthreshold, nlines, verbose, scale)
    return linepar, acc

