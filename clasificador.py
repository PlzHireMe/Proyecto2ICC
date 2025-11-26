import cv2
import glob
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
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
nombres_imagenes = []

#Trabajar Imagenes
#Adriano

for archivo in glob.glob("imagenes/*.*"):
    img = cv2.imread(archivo, cv2.IMREAD_GRAYSCALE)
    nombres_imagenes.append(archivo)
    img = cortar_imagenes(img)
    img = cv2.resize(img, (8, 8))
    for i in range(8):
        for j in range(8):
            img[i][j] = (255 - img[i][j])/255*16
    imagenes_clasificar.append(img.flatten())

#Clasificar
#Matias

digitos = datasets.load_digits()
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(digitos.data, digitos.target)

distancias, indices = KNN.kneighbors(imagenes_clasificar, n_neighbors=3)
#Indices es una matriz N x 3

'''e) Para cada número nuevo, imprima los targets que corresponden los 3 dígitos más
parecidos a él. Además, indique qué número es realmente. Así podremos tener en
pantalla su etiqueta verdadera, y los target de los 3 vecinos más cercanos./'''

'''f) Intente clasificar a sus nuevos dígitos:'''


for i in range(len(imagenes_clasificar)):
    print("Imagen", nombres_imagenes[i][9:10], "tiene como targets:", digitos.target[indices[i]])
    
    if digitos.target[indices[i]][0] == digitos.target[indices[i]][1] or digitos.target[indices[i]][0] == digitos.target[indices[i]][2]:
        X = digitos.target[indices[i]][0]
    elif digitos.target[indices[i]][1] == digitos.target[indices[i]][2]:
        X = digitos.target[indices[i]][1]
        
    #Si no, escogemos al de menor distancia
    indice_menor_distancia = np.argmin(distancias[i])
    X = digitos.target[indices[i][indice_menor_distancia]]

    
    print('"Soy la inteligencia artificial, y he detectado que el dígito ingresado corresponde al número', X, '”, donde', X, 'es un número entre 0 y 9.')

#Matriz Confusion
#Alejandro









