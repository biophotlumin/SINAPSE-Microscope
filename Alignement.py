#!/usr/bin/env python
# coding: utf-8

'''library'''
import ALPlib

import matplotlib.pyplot as plt
import numpy as np
import imageio
from pypylon import pylon
from pypylon import genicam
import time
from scipy import optimize
import scipy
import sys
from aotools.functions import phaseFromZernikes
import sinapse

'''Variable initialistaion'''
npy=1024 #number of pixel in x direction
npx=768 # number of pixels in y  direction
d=13.6e-6 # size of one pixel in meter
T_porteuse=8*d
sigma_beam=5e-3 # width of the beam incident on the DMD in meter
lam =1.028e-6 #Longueure d'onde en METRE
corr="C:\\Users\\lac\\Desktop\\Florian\\git\\SINAPSE-Microscope\\Phase correction\\PhaseMask.npy"
q=1


DMD = ALPlib.ALP4(version = '4.3', libDir = 'C:/Program Files/ALP-4.3/ALP-4.3 API')
DMD.Initialize()

def disk_mask(shape, radius, center = None):
    '''
    Generate a binary mask with value 1 inside a disk, 0 elsewhere
    :shape: list of integer, shape of the returned array
    :radius: integer, radius of the disk
    :center: list of integers, position of the center
    :return: numpy array, the resulting binary mask
    '''
    if not center:
        center = (shape[1]//2,shape[0]//2)
    X,Y = np.meshgrid(np.arange(shape[0]),np.arange(shape[1]))
    mask = (Y-center[0])**2+(X-center[1])**2 < radius**2
    return mask.astype(int)


'''Main'''
#_____première imag servant à reperer le centre du DMD
center=np.zeros([768,1024],dtype=int)+disk_mask([1024,768],16)*int(255)
center=np.where(center==0,255,0)

plt.imshow(center)
plt.colorbar()
sinapse.afficher(center,DMD,100000)
plt.show()



ref_position=sinapse.Holo_Position(0,0,0,corr,T_porteuse,q,lam)
sinapse.afficher(ref_position,DMD,10000)
plt.imshow(ref_position)
plt.show()
DMD.Halt()




'''sequence=[]

nt=11
for i in range(1):
    for j in range(nt):
        sequence.append(holo(lam,0,-5+j,0,corr,T_porteuse).ravel())
sequence=concatenate([i for i in sequence])
DMD.SeqAlloc(nbImg = nt*1, bitDepth = 1)
DMD.SeqPut(sequence)
DMD.SetTiming(illuminationTime=int(1e5))
DMD.Run(loop=True)
plt.imshow(ref_position)
plt.show()'''
