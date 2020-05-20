#!/usr/bin/env python
# coding: utf-8

'''library'''
import ALPlib
import cv2
import matplotlib.pyplot as plt
from numpy import *
import imageio
from pypylon import pylon
from pypylon import genicam
import time
from scipy import optimize
import scipy
import sys
from aotools.functions import phaseFromZernikes

'''Variable initialistaion'''
npy=1024 #number of pixel in x direction
npx=768 # number of pixels in y  direction
d=13.6e-6 # size of one pixel in meter

sigma_beam=5e-3 # width of the beam incident on the DMD in meter
lam =0.633e-6 #Longueure d'onde en METRE
DMD = ALPlib.ALP4(version = '4.3', libDir = 'C:/Program Files/ALP-4.3/ALP-4.3 API')
DMD.Initialize()
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
'''function definition'''

def capture():
    '''capture an image with the basler camera'''
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

def holo(wavelentgh,x_ech,y_ech,z_ech,corr):
    '''generate an hologramm to focus the laser beam at the coordinates (x_ech,y_ech,z_ech) in the referential of the set up
    :wavelength : wavelangth of the excitation
    :x,y,z_ech : coordinates of the desired focus
    :corr : complex array of the pahse correction qpplied to the DMD bfore generation of holo '''
    d=13.6e-6 #Taille d'un coté des micromirroir en METRE
    theta=12*pi/180 # Angle que fait le vecteur d'onde de l'onde de reference avec l'axe z, normal au plan moyen du dmd.
    rho=45*pi/180
    r=1
    #_________déterminons phi, l'angle de la rotation qui relie R à Ro_____________________
    s=1/sqrt(2)
    cp=sqrt(cos(theta)**2-sin(theta)*r*wavelentgh*cos(rho)/(2*d)-(r*wavelentgh/(4*d))**2) #cp=cos(phi)
    pc=arccos(cp)*180/pi #pc=phi juste pour avoir une idée de sa valeur
    sp=sqrt(sin(theta)**2+sin(theta)*r*wavelentgh*cos(rho)/(2*d)+(r*wavelentgh/(4*d))**2)
    ps=arcsin(sp)*180/pi
    phi=arctan(sp/cp)*180/pi
    nx=(-r*wavelentgh/(4*d)-sin(theta)*cos(rho))/sp
    ny=(sin(theta)*sin(rho))/sp
    nz=0

    n=array([ [nx],  [ny], [nz] ]) # vecteur dirigeant l'axe de la rotation pour passer du repèreDMD à repère AxeOptique
    I=array([ [1,0,0],[0,1,0], [0,0,1] ])
    R1=array([[nx**2, nx*ny, nx*nz], [ny*nx, ny**2, ny*nz],[nz*nx, ny*nz, nz**2] ])
    R2=array([ [0, -nz, ny], [ nz, 0, -nx],[-ny, nx, 0] ])

    Npx,Npy=768,1024 #Nombre de pixels sur le DMD
    dmd=zeros((Npx,Npy))
    i0,j0=Npx/2,Npy/2 #Indice du pixel central
    f1,f2,fobj=0.5,0.3,-0.00333 #focales en METRE des lentilles resp. L1,L2,Objectif
    F=f1*fobj/f2
    neau=1.33
    #_________________________________________________________________________________
    xe=x_ech*1e-6
    ye=y_ech*1e-6
    ze=z_ech*1e-6
    #on indice avec un O les coordonnées des points conjuguées dans le repère de la lentille de focal F
    if ze==0:
        zco=-10000
        ze=(neau*(f1*fobj/f2)**2)/-zco
    else:
        zco=-(neau*(f1*fobj/f2)**2)/ze

    #determinons les coordonnées x2,y2 de l'image de xe,ye par Lobj
    z2=-neau*fobj**2/ze
    x2=neau*xe*(fobj+z2)/(-neau*fobj+ze)
    y2=neau*ye*(fobj+z2)/(-neau*fobj+ze)

    #determinons les coordonnées x1,y1 de l'image de x2,y2 par L2
    z1=-f2**2/z2
    x1=x2*(-f2+z1)/(f2+z2)
    y1=y2*(-f2+z1)/(f2+z2)

    #determinons les coordonnées xco,yco de l'image de x1,y1 par L1
    zco=-f1**2/z1
    xco=x1*(-f1+zco)/(f1+z1)
    yco=y1*(-f1+zco)/(f1+z1)
    #_______________passage du repère AxeOptique vers DMD____________________________

    Co=array([[xco],[yco],[zco]])
    R=I*cp+(1-cp)*R1+sp*R2
    C=dot(R,Co)
    #______________Determination de l'interferogramme sur le DMD_____________________
    sr=sin(theta)*sin(rho)
    cr=sin(theta)*sin(rho)
    # Fonction génératrice numpy

    def f(i, j):
        x=(i+1/2-i0)*d
        y=(j+1/2-j0)*d
        return ((1+cos((2*pi/wavelentgh)*(sqrt((C[0][0]-x)**2+(C[1][0]-y)**2+C[2][0]**2)-sr*x-cr*y)-corr[i,j]))>1)
        #return ((1+cos((2*pi/wavelentgh)*(sqrt((C[0][0]-x)**2+(C[1][0]-y)**2+C[2][0]**2)-sr*x-cr*y)))>1)

    # Génération parallélisée de l'array avec fromfunction et la fonction génératrice :)
    #filepath = "C:\\Users\\lac\\Desktop\\Florian\\Alignement\\"
    #file_name = filepath+str(m)+".npy"
    dmd=where(fromfunction(f, (Npx, Npy), dtype=int),0,255)

    #dmd=where(fromfunction(f, (Npx, Npy), dtype=uint8),0x00,0x80)
    #dmd=fromfunction(f, (Npx, Npy), dtype=uint8)
    return(dmd)

def get_disk_mask(shape, radius, center = None):
    '''
    Generate a binary mask with value 1 inside a disk, 0 elsewhere
    :param shape: list of integer, shape of the returned array
    :radius: integer, radius of the disk
    :center: list of integers, position of the center
    :return: numpy array, the resulting binary mask
    '''
    if not center:
        center = (shape[0]//2,shape[1]//2)
    X,Y = meshgrid(arange(shape[0]),arange(shape[1]))
    mask = (Y-center[0])**2+(X-center[1])**2 < radius**2
    return mask.astype(int)

def complex_mask_from_zernike_coeff(shape, radius, center, vec):
    '''
    Generate a complex phase mask from a vector containting the coefficient of the first Zernike polynoms.
    :param DMD_resolution: list of integers, contains the resolution of the DMD, e.g. [1920,1200]
    :param: integer, radius of the illumination disk on the DMD
    :center: list of integers, contains the position of the center of the illumination disk
    :vec: list of float, the coefficient of the first Zernike polynoms
    '''
    # Generate a complex phase mask from the coefficients
    zern_mask = exp(1j*phaseFromZernikes(vec,2*radius))
    # We want the amplitude to be 0 outside the disk, we fist generate a binary disk mask
    amp_mask = get_disk_mask([2*radius]*2,radius)
    # put the Zernik mask at the right position and multiply by the disk mask
    mask = zeros(shape = shape, dtype=complex)
    mask[center[0]-radius:center[0]+radius,
         center[1]-radius:center[1]+radius] = zern_mask*amp_mask
    return mask

def get_cost(img,mask_center, mask_radius = 8):
    res = img.shape
    X,Y = meshgrid(arange(res[1]),arange(res[0]))
    # We generate a mask representing the disk we want the intensity to be concentrated in
    mask = (X-mask_center[0])**2+(Y-mask_center[1])**2 < mask_radius**2
    signal = sum(img)/sum(mask)
    plt.imshow(img)
    noise = sum((img)*(1.-mask))/sum(1.-mask)
    cost = signal/noise
    return cost

def afficher(img,DMD):
    '''display holo on the DMD'''
    # Allocate the onboard memory for the image sequence
    DMD.SeqAlloc(nbImg = 1, bitDepth = 1)
    # Send the image sequence as a 1D list/array/numpy array
    DMD.SeqPut(imgData = img.ravel())
    # Set image rate to 50 Hz
    DMD.SetTiming(illuminationTime = 1000000)
    # Run the sequence in an infinite loop
    DMD.Run()
    return()

'''Main'''
COUT=[]
COEFF=[]
for test in range (1): #On test chaque coefficients entre -1 et 1 avec pas 0.05

    complex_mask = complex_mask_from_zernike_coeff(shape = [npx,npy],
                                           radius = int(sigma_beam/d),
                                           center = [npx//2,npy//2],
                                           vec = [0.,1,-0.1,-0.45,0.4,-0.70,-0.6,0.7,0.25])
    basedmd=holo(lam,0,0,0,angle(complex_mask))
    afficher(basedmd,DMD)
    lum=capture()
    c=get_cost(lum,[737,653])
    cz=-1+0.05*test
    COUT.append(c)
    COEFF.append(cz)


plt.show()
plt.imshow(angle(basedmd))
plt.colorbar()
plt.show()
