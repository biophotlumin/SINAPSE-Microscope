#!/usr/bin/env python
# coding: utf-8

'''Code principal'''
import ALPlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imageio
from pypylon import pylon
from pypylon import genicam
import time
from scipy import optimize
import scipy
import sys

#____________________aquisition Basler________________________

# connexion à la première caméra Basler reconnue
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

#___________________initialisation_DMD________________________

# charger le Vialux .dll
DMD = ALPlib.ALP4(version = '4.3', libDir = 'C:/Program Files/ALP-4.3/ALP-4.3 API')
DMD.Initialize()

#____________________enregistrer_image_sous_forme_d'un_array_______________________
def capture():
    # Number of images to be grabbed.
    countOfImagesToGrab = 1

    # The exit code of the sample application.
    exitCode = 0

    # Create an instant camera object with the camera device found first.
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    # Print the model name of the camera.
    print("Using device ", camera.GetDeviceInfo().GetModelName())

    # The parameter MaxNumBuffer can be used to control the count of buffers
    # allocated for grabbing. The default value of this parameter is 10.
    camera.MaxNumBuffer = 5

    # Start the grabbing of c_countOfImagesToGrab images.
    # The camera device is parameterized with a default configuration which
    # sets up free-running continuous acquisition.
    camera.StartGrabbingMax(countOfImagesToGrab)

    # Camera.StopGrabbing() is called automatically by the RetrieveResult() method
    # when c_countOfImagesToGrab images have been retrieved.
    while camera.IsGrabbing():
        # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        # Image grabbed successfully?
        if grabResult.GrabSucceeded():
            # Access the image data.
            print("SizeX: ", grabResult.Width)
            print("SizeY: ", grabResult.Height)
            img = grabResult.Array
            #print("Gray value of first pixel: ", img[xmax, ymax])
        else:
            print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
        grabResult.Release()
    return(img)
#____________________afficher_sur_DMD________________________

def afficher(img,DMD):
    # Allocate the onboard memory for the image sequence
    DMD.SeqAlloc(nbImg = 1, bitDepth = 1)
    # Send the image sequence as a 1D list/array/numpy array
    DMD.SeqPut(imgData = img.ravel())
    # Set image rate to 50 Hz
    DMD.SetTiming(illuminationTime = 1000000)
    # Run the sequence in an infinite loop
    DMD.Run()
    return()

#____________________fonction_à_ajuster________________________
def f_inter(x,phi,I,K,c):

    wavel=0.633
    foc=300000*3333/500000
    d=13.6

    #w=1000 #essayer d'obtenir w automatiquement from the cardinal fit
    dr=d*dm*np.sqrt((k-nm//2)**2+(l-nm//2)**2)

    return(K+I*np.exp(-(x-ymax*6.45)**2/(2*(170)**2))*(1+c*np.cos(phi+2*np.pi*dr*(x-ymax*6.45)/(60*wavel*foc))))
    #return(K+I*np.exp(-(x-ymax*6.45)**2/(2*(170)**2))*(1+c*np.cos(phi+2*np.pi*dr*(x-ymax*6.45)/(wavel*foc))))


def cardinal(height, center_x, center_y, width):
    """Returns a gaussian function with the given parameters"""
    width = float(width)

    return lambda x,y: height*np.sinc(((center_x-x)+(center_y-y))/(np.sqrt(2)*width))*np.sinc(((center_x-x)-(center_y-y))/(np.sqrt(2)*width))

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    width=(width_x+width_y)/2
    return height, x, y, width

def fitcardinal(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params =73, 548, 766,  59 #moments(data)
    errorfunction = lambda p: np.ravel(cardinal(*p)(*np.indices(data.shape))-data)
    p, success = scipy.optimize.leastsq(errorfunction, params)
    return p


#____________________trouver_le_profil________________________
'''xm,ym sont determinés suite au fit sincardinal'''
def profil(img,xm,ym):
    l=[]
    for i in range(400):
        s=0
        for j in range(100):
            s+=img[ym-200+i,xm-50+j]
        l.append([(ym-200+i)*6.45,s])
    return(l)

#____________________balayage_Mpxs____________________________



dm=32 #nombre de pixels sur un côté d'un macropixel
nm=768//dm #nombre de macropixels sur lesquels on fait la mesure
print(nm)
n=0

phi=np.zeros((nm,nm)) #on initialise le tableau qui contiendra les mesure de phase
filepath = "C:\\Users\\lac\\Desktop\\Florian\\phasecorrection\\"
background=capture()
basedmd=np.zeros((768,1024),dtype=int)
testinter=np.zeros((768,1024),dtype=int) #initialisation du tableau dans lequel on entrera les images DMD pour interferences
#On cree la fonction reflectence pour la reference de phase
for i in range(dm//8):
    for j in range(dm):
        basedmd[(nm//2)*dm+j,128+(nm//2)*dm+8*i]=255
        basedmd[(nm//2)*dm+j,128+(nm//2)*dm+8*i+1]=255
        basedmd[(nm//2)*dm+j,128+(nm//2)*dm+8*i+2]=255
        basedmd[(nm//2)*dm+j,128+(nm//2)*dm+8*i+3]=255
afficher(basedmd,DMD)
plt.imshow(basedmd)
plt.show()
# basedmd_measured=capture() #on mesure la figure d'éclairement dans pf L1 pour trouver le point de focalisation
#
# #for i in range(768):
#  #   for j in range(1024):
#   #     if basedmd_measured[i,j]>background[i,j]:
#    #         basedmd_measured[i,j]=basedmd_measured[i,j]-background[i,j]
#     #    else:
#      #       basedmd_measured[i,j]=0
# image = np.floor(basedmd)
# image = image.astype(np.uint8)
# imageio.imwrite(filepath+"centrer.png", image)
# DMD.Halt()
# DMD.FreeSeq()
# parametres=fitcardinal(basedmd_measured)
# print(parametres)
# X,Y = np.indices(basedmd_measured.shape)
# fit = cardinal(*parametres)
#
# plt.contour(fit(*np.indices(basedmd_measured.shape)), cmap=plt.cm.copper)
# plt.imshow(basedmd_measured),plt.colorbar()
# plt.show()
#
#
# xmax=int(parametres[2]+1/2)
# ymax=int(parametres[1]+1/2)
# #xmax=568
# #ymax=761
#
# #_____________on_applique_les_reflectences_trous_d'Young_et_mesurer_dephasage____________________________
# for k in range(768//dm):
# #for k in range(7,8):
#     for l in range(768//dm):
#     #for l in range(10,14):
#         testinter=np.zeros((768,1024),dtype=int)
#         print(k,l)
#         for i in range (dm//4):
#             for j in range (dm):
#                 testinter[k*dm+j,128+l*dm+4*i]=255
#                 testinter[k*dm+j, 128+l*dm+4*i+1]=255
#         if k==(nm//2)  and l==(nm//2):
#             phi[k,l]=0
#         else:
#             if (l-nm//2)!=0:
#                 testinter=testinter+basedmd
#                 theta=np.arctan((k-nm//2)/(l-nm//2))*180/np.pi #on determine l'angle duquel faire tourner l'image aquise pour pouvoir avoir un profil selon une colonne
#                 rotation_matrix=cv2.getRotationMatrix2D((xmax,ymax),theta+45+90,1) #on calcul la matrice de rotation
#                 afficher(testinter,DMD)
#                 testinter_measured=capture()
#                 DMD.Halt()
#                 DMD.FreeSeq()
#                 testinter_rotation=cv2.warpAffine(testinter_measured,rotation_matrix,(1392,1040))#on effectue la rotation
#
#                 inter=profil(testinter_rotation,xmax,ymax)#on trace le profil des interferences
#                 xdata=[inter[i][0] for i in range(400)]
#                 ydata=[inter[i][1] for i in range(400)]
#                 p00 = [0,max(ydata),200,0.5] #paramètres initiaux du fit
#                 print(p00)
#                 popt, pcov = scipy.optimize.curve_fit(f_inter, xdata, ydata,p0=p00,bounds=([-np.pi,0,0,0],[np.pi,max(ydata)*1.5,1500,1]))
#                 print(popt)
#                 plt.plot(xdata,ydata)
#                 yfit=[f_inter(xdata[i],popt[0],popt[1],popt[2],popt[3]) for i in range(400)]
#                 #yth=[f_inter(xdata[i],0,15000,500,0.8) for i in range(400)]
#                 #plt.plot(xdata,yth)
#                 plt.plot(xdata,yfit)
#                 plt.show()
#                 phi[k,l]=popt[0]
#                 plt.imshow(testinter_rotation),plt.colorbar()
#                 plt.show()
#
#             else:
#                 testinter=testinter+basedmd
#                 theta=45
#                 rotation_matrix=cv2.getRotationMatrix2D((xmax,ymax),theta,1)
#                 afficher(testinter,DMD)
#                 testinter_measured=capture()
#                 DMD.Halt()
#                 DMD.FreeSeq()
#                 testinter_rotation=cv2.warpAffine(testinter_measured,rotation_matrix,(1392,1040))
#                 inter=profil(testinter_rotation,xmax,ymax)
#                 xdata=[inter[i][0] for i in range(400)]
#                 ydata=[inter[i][1] for i in range(400)]
#                 p00 = [0,max(ydata),200,0.5]
#                 popt, pcov = scipy.optimize.curve_fit(f_inter, xdata, ydata,p0=p00,bounds=([-np.pi,0,0,0],[np.pi,max(ydata)*1.5,1500,1]))
#                 '''plt.plot(xdata,ydata)
#                 yfit=[f_inter(xdata[i],popt[0],popt[1],popt[2],popt[3]) for i in range(400)]
#                 plt.plot(xdata,yfit)
#                 plt.show()'''
#                 phi[k,l]=popt[0]
#                 '''plt.imshow(testinter_rotation),plt.colorbar()
#                 plt.show()'''
# DMD.Free()
#
#
#
# # In[26]:
#
#
# plt.imshow(phi),plt.colorbar()
# plt.show()
#
#
# # In[27]:
#
#
# #______________On redimensionne le mapping de la phase au tailles du DMD________________________
# phasemask=np.zeros((768,1024))
# for i in range(768//dm):
#     for j in range(768//dm):
#         for k in range(dm):
#             for l in range(dm):
#                 phasemask[i*dm+k,128+j*dm+l]=phi[i,j]
#
# plt.imshow(phasemask),plt.colorbar()
# plt.show()
#
#
# # In[28]:
#
#
# np.save('phasemask_objectif',phasemask)
