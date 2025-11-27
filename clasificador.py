import cv2
import glob
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import numpy as np

#Dylan
def cortar_imagenes(img):
    #Blanco > 190
    arriba = 0
    abajo = 0
    izquierda = 0
    derecha = 0
    for fila in img:
        if np.all(fila > 190): 
            arriba += 1
        else: 
            break
    for fila in img[::-1]:
        if np.all(fila > 190):
            abajo += 1
        else: 
            break
    for columna in img.T:
        if np.all(columna > 190):
            izquierda += 1
        else: 
            break
    for columna in img.T[::-1]:
        if np.all(columna > 190):
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

def calcularDistancia(lista, listaB):
    suma = 0
    for i in range(len(lista)):
        suma += (listaB[i] - lista[i])**2
    return round(suma**0.5, 3)

def KNearestNeighbors(datos_clasificar, datos, etiquetas, nVecinos):
    distancias = []
    valores = []

    for dato_clasificar in datos_clasificar:
        distancia = []
        distancia_sort = []
        valor = []
        for dato in datos:
            distancia.append(calcularDistancia(dato_clasificar, dato))
        indices = np.argsort(distancia)

        for i in range(nVecinos):
            distancia_sort.append(float(distancia[indices[i]]))
            valor.append(int(etiquetas[indices[i]]))
        distancias.append(distancia_sort)
        valores.append(valor)
        
    return (distancias, valores)

def predict(etiquetas):
    pred = Counter(etiquetas)
    return pred.most_common(1)[0][0]

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
#Dylan

digitos = datasets.load_digits()

distancia, valores = KNearestNeighbors(imagenes_clasificar, digitos.data, digitos.target, 3)
#Indices es una matriz N x 3

for i in range(len(imagenes_clasificar)):
    print("Imagen", nombres_imagenes[i][9:10], "tiene como targets:", valores[i])
    X = predict(valores[i])
    print('"Soy la inteligencia artificial, y he detectado que el dígito ingresado corresponde al número', X, '”, donde', X, 'es un número entre 0 y 9.')


#Matriz Confusion
#Alejandro





