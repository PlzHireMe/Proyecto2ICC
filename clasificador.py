import cv2
import glob
from sklearn import datasets
import numpy as np

#Dylan
def cortar_imagenes(img):
    #Blanco > 150
    arriba = 0
    abajo = 0
    izquierda = 0
    derecha = 0
    for fila in img:
        if np.all(fila > 150):
            arriba += 1
        else: 
            break
    for fila in img[::-1]:
        if np.all(fila > 150):
            abajo += 1
        else: 
            break
    for columna in img.T:
        if np.all(columna > 150):
            izquierda += 1
        else: 
            break
    for columna in img.T[::-1]:
        if np.all(columna > 150):
            derecha += 1
        else: 
            break

    img = img[arriba : img.shape[0] - abajo, izquierda : img.shape[1] - derecha]
    alto, ancho = img.shape[:2]
    lado = max(alto, ancho)
    pad_alto = lado - alto
    pad_ancho = lado - ancho

    pad_arriba = pad_alto // 2
    pad_abajo = pad_alto - pad_arriba 

    pad_izq = pad_ancho // 2
    pad_der = pad_ancho - pad_izq      

    img = np.pad( img,  ((pad_arriba, pad_abajo), (pad_izq, pad_der)),  mode='constant',  constant_values=255)
    return img

imagenes_clasificar = []

for archivo in glob.glob("imagenes/*.*"):
    img = cv2.imread(archivo, cv2.IMREAD_GRAYSCALE)
    img = cortar_imagenes(img)
    img = cv2.resize(img, (8, 8))
    for i in range(8):
        for j in range(8):
            img[i][j] = (255 - img[i][j])/255*16
    imagenes_clasificar.append(img)
