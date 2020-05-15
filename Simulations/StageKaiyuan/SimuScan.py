#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


d=101 #resolution du scan en nanomètre
npx=25 # nombre de pixel du scan
picture=np.zeros([npx,npx]) # fonction numpy qui initie un array de taille npx² avec des zeros
sigma=250 # largeur caractéristique de la gaussienne
nbspot=1 # nombe de particules dans le champs simulé
#juste un test pour tester le double remote
'''___un petit morceau de code pour comprendre ce que signifie les indieces en numpy array_____'''
'''picturei=np.zeros([npx,npx]) # pour voir quelle direction est influée par u ne modification du premierindice du tableau
for i in range(npx):
    picturei[i,0:]=i
picturej=np.zeros([npx,npx]) # pour voir quelle direction est influée par u ne modification du premierindice du tableau
for j in range(npx):
    picturej[0:,j]=j
fig=plt.figure(figsize=(10,20))

plt.subplot(121)
plt.imshow(picturei)

plt.xlabel("variation du  premier indice d'un tableau python")
plt.subplot(122)
plt.imshow(picturej)
plt.xlabel("variation du  duxième indice d'un tableau python")
plt.show()'''
'''_____________________________________________________________________________________________'''


# In[7]:


'''_________On ajoute les images des particules au tableau________'''
tailleimage=6*int(sigma/d) # sigma/d correspond converti l'ecart des gaussiennes en pixel. tailleimage est le nombre de pixel que l'on utilise pour simuler une mage
picture=np.zeros([npx,npx]) # on reinitialise le tableau
for i in picture : # bruit de fond
    i=np.random.normal(60,np.sqrt(60)) #60 est l'intensité moyenne du bruit de fond

for n in range (nbspot): #boucle for qui va ajouter chaque image de particules
    In=np.random.normal(500,200) # on tire aléatoirement l'intensité et la position des particules
    xn=np.random.randint(12,npx-12)
    yn=np.random.randint(12,npx-12)
    for gx in range(tailleimage):
        for gy in range(tailleimage):
            picture[xn+(gx-tailleimage//2),yn+(gy-tailleimage//2)]+=In*np.exp(-(((gx-tailleimage//2)*d)**2+((gy-tailleimage//2)*d)**2)/(2*sigma)**2)

fig=plt.figure(figsize=(10,10))
plt.imshow(picture)
plt.title("scan simulé")
plt.xlabel("un pixel= "+str(d)+" nm")
plt.colorbar()
plt.show()


# In[ ]:
