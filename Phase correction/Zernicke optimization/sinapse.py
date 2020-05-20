""" Module_microscope_SINAPSE Florian Semmer """
import ALPlib
from numpy import *
import imageio
from pylab import *
from time import time
from nifpga import Session
from pypylon import pylon
from pypylon import genicam



#_________________________CameraBasler________________________________________________
def capture():
    # nombre d'image à enregistrer
    countOfImagesToGrab = 1

    # code sortie de l'application.
    exitCode = 0

    #.On cree un objet camera attaché à la première caméra trouvée
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    # on affiche le modèle de la caméra.
    print("Using device ", camera.GetDeviceInfo().GetModelName())

    # noombre de buffers autorisé pour la capture d'images
    camera.MaxNumBuffer = 5

    # images.commence l'aquisition d'une image
    # paramètres par defaut
    #aquisition continue.
    camera.StartGrabbingMax(countOfImagesToGrab)

    while camera.IsGrabbing():
        # attendre maximum 5000ms pour une image.
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            # on affuche les données de l'image
            print("SizeX: ", grabResult.Width)
            print("SizeY: ", grabResult.Height)
            img = grabResult.Array
            print("Gray value of first pixel: ", img[xmax, ymax])
        else:
            print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
        grabResult.Release()
    return(img)

#____________________DMD_____________________________________________________________

def afficher(img,DMD):
    # on prépare la mémoire sur la ram du DMD pour n tableau de bits
    DMD.SeqAlloc(nbImg = 1, bitDepth = 1)
    # envoyer la séquence à afficher sur la RAM
    DMD.SeqPut(imgData = img.ravel())
    # parametrer l'affichage de la dernière séquence chargée
    DMD.SetTiming(illuminationTime = 1000000)
    # afficher la séquence dans une boucle infinie
    DMD.Run()
    return()

def Holo_Position(x_ech,y_ech,z_ech,correction_filepath):
    '''x, y et z sont les 3 coordonnées (micromètre) du point de focalisation du laser. La fonction renvoie alors un tableau de uint8 étant l'hologramme à afficher'''

    wavelentgh =0.633e-6 #Longueure d'onde en METRE
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
    phi=np.load(correction_filepath)
    def f(i, j):
        x=(i+1/2-i0)*d
        y=(j+1/2-j0)*d
        return ((1+cos((2*pi/wavelentgh)*(sqrt((C[0][0]-x)**2+(C[1][0]-y)**2+C[2][0]**2)-sr*x-cr*y)-phi[i,j]))>1)
        #return ((1+cos((2*pi/wavelentgh)*(sqrt((C[0][0]-x)**2+(C[1][0]-y)**2+C[2][0]**2)-sr*x-cr*y)))>1)

    # Génération parallélisée de l'array avec np.fromfunction et la fonction génératrice :)
    #filepath = "C:\\Users\\lac\\Desktop\\Florian\\Alignement\\"
    #file_name = filepath+str(m)+".npy"
    dmd=where(fromfunction(f, (Npx, Npy), dtype=int),0,255)

    #dmd=where(fromfunction(f, (Npx, Npy), dtype=uint8),0x00,0x80)
    #dmd=fromfunction(f, (Npx, Npy), dtype=uint8)
    return(dmd)


def Phase_correction():
    return()

def Raster_Scan(Xpix,Ypix,xinit,yinit,zinit,pictime,dx,dy):
    pici=zeros((Xpix,Ypix))
    pict=zeros((Xpix,Ypix))
    with Session("C:\\Users\\lac\\Desktop\\Florian\\FPGA\\TestFPGA\\scan_imaging\\FPGA Bitfiles\\scanimaging_FPGATarget_fpgamain_BYS4O-sfyPE.lvbitx","RIO0") as session :
        count = session.fifos['count to PC']
        it = session.fifos['integration time to PC']
        count.start()
        it.start()
        for m in range(Xpix):
            for n in range(Ypix):
                l.append(Holo_Position(xinit+Xpix*mdx,yinit+Ypix*dy,zinit,correction_mask).ravel())
        adresses.append(DMD.SeqAlloc(nbImg =Xpix*Ypix, bitDepth = 1))
        imgSeq=concatenate([i for i in l])  #on met les tableaux 1D les uns à la suite des autres pour créer une séquence
        DMD.SeqPut(imgSeq,SequenceId=adresses[0])
        DMD.SetTiming(SequenceId = adresses[0], pictureTime = pictime) #picture time est le temps écoulé entre deux images d'une séquence en microsecondes
        DMD.Run(SequenceId=adresses[0],loop=False)
        #session.reset()
        #session.run()
        DMD.Wait()
        pc=count.read(Xpix*Ypix-1,10) #nombre de valeur à lire dans le tampon et temps maximum passé à tester la presence d'une valeur dans le tampon
        itime=it.read(Xpix*Ypix-1,10)
        count.stop() #On arrète les tampons
        it.stop()
        pc.data.append(0) #La méthode .read retourne une liste des valeurs lu appelée "data" et le nombre d'élements encore présents dans le tampon. Ici on ajoute un zéro à la data car la première valeure lors de la mesure n'a pa été prise en compte
        itime.data.append(0)

    for i in range(Xpix):
        for j in range(Ypix):
            pici[i][j]= pc.data[Ypix*i+j] # On fait de la liste data une image
            pict[i][j]= itime.data[Ypix*i+j]
    return(pici/pict)


def to_c_array(values, ctype="uchar", name="table", formatter=str):
    # apply formatting to each element
    colcount=len(values)
    values = [formatter(v) for v in values]

    # split into rows with up to `colcount` elements per row
    rows = [values[i:i+colcount] for i in range(0, len(values), colcount)]

    # separate elements with commas, separate rows with newlines
    body = ',\n    '.join([', '.join(r) for r in rows])

    # assemble components into the complete string
    return '{} {}[] = {{\n    {}}};'.format(ctype, name, body)
