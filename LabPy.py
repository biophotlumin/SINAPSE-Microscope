#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sinapse
import numpy as np
import ALPlib
import array
from ctypes import *
from nifpga import Session
import matplotlib.pyplot as plt
import trackpy as tp
import time


# In[14]:

def displayref():
    Itref = 1000
    xpix=768
    ypix=1024
    T_porteuse=8*13.6e-6
    corr="C:\\Users\\lac\\Desktop\\Florian\\git\\SINAPSE-Microscope\\Phase correction\\PhaseMask.npy"
    l=[sinapse.Holo_Position(0,0,0,corr,T_porteuse).ravel()]
    l=np.concatenate([i for i in l])

    aref = (c_ubyte*(xpix*ypix))()
    for i,pix in enumerate(l):
        aref[i]=pix
    global freezeDMD
    freezeDMD = ALPlib.ALP4(version = '4.3', libDir = 'C:/Program Files/ALP-4.3/ALP-4.3 API')
    freezeDMD.Initialize()
    freezeDMD.SeqAlloc(nbImg = 1, bitDepth = 1)
    freezeDMD.SeqPut(aref,dataFormat='C')
    freezeDMD.SetTiming(illuminationTime=int(Itref))
    freezeDMD.Run(loop=True)


    return('OK')

def shutdownref():
    freezeDMD.Halt()
    freezeDMD.Free()
    return('OK')

def bibgen(nx,ny):
    global a
    xpix=768
    ypix=1024
    global nseq
    nseq=nx*ny+1
    T_porteuse=8*13.6e-6
    global It
    It=1e3#illumination time en microseconde
    corr="C:\\Users\\lac\\Desktop\\Florian\\git\\SINAPSE-Microscope\\Phase correction\\PhaseMask.npy"

    l=[sinapse.Holo_Position(0,0,0,corr,T_porteuse).ravel()]

    X=np.linspace(-5,5,nx)
    print(X)
    Y=np.linspace(-5,5,ny)



    for ix in range(nx):
        cx=X[ix]
        for iy in range(ny):
            cy=Y[iy]
            l.append(sinapse.Holo_Position(cx,cy,0,corr,T_porteuse).ravel())
            print(100*(ix*ny+iy)/nseq,'%')

    print('5fps c converting ... ')

    l=np.concatenate([i for i in l])
    ntot= nseq*xpix*ypix
    a = (c_ubyte *ntot)()
    for i,pix in enumerate(l):
        a[i]=pix
    print(a)
    return ('OK')

def scan(nx,ny):
    print(a)
    DMD = ALPlib.ALP4(version = '4.3', libDir = 'C:/Program Files/ALP-4.3/ALP-4.3 API')
    DMD.Initialize()
    DMD.SeqAlloc(nbImg = nseq, bitDepth = 1)
    DMD.SeqPut(a,dataFormat='C')
    DMD.SetTiming(illuminationTime=int(It))


    with Session("C:\\Users\\lac\\Desktop\\Florian\\FPGA\\holoscan\\FPGA Bitfiles\\holoscan_FPGATarget_fpgamain_FGl-dvEuVeU.lvbitx","RIO0") as session :
        count = session.fifos['count to PC']
        it = session.fifos['integration time to PC']
        count.start()
        it.start()
        session.reset()
        session.run()
        DMD.Run(loop=False)

        DMD.Wait()
        DMD.Free()
        pc=count.read(nx*ny,200)
        itime=it.read(nx*ny,200)
        count.stop()
        it.stop()

        pici=np.zeros((nx,ny))
        pict=np.zeros((nx,ny))
        pic=np.zeros((nx,ny))
        for i in range(nx):
                for j in range(ny):
                    pici[i][j]= pc.data[j+nx*i]/(It*1e-6)
                    pict[i][j]= itime.data[j+nx*i]
    return(pici.tolist())
