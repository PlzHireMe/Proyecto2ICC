import cv2
import glob
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from collections import Counter
import numpy as np

#Dylan

def calcularDistancia(lista, listaB):
    suma = 0
    for i in range(len(lista)):
        suma += (listaB[i] - lista[i])**2
    return suma**0.5

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

Reales = []
Predecidos = []

for i in range(len(imagenes_clasificar)):
    print("Imagen", nombres_imagenes[i][9:10], "tiene como targets:", valores[i])
    X = predict(valores[i])
    Reales.append(int(nombres_imagenes[i][9:10]))
    Predecidos.append(X)
    print('"Soy la inteligencia artificial, y he detectado que el dígito ingresado corresponde al número', X, '”, donde', X, 'es un número entre 0 y 9.')


#Matriz Confusion
#Alejandro

MatrizConfusion10Class = confusion_matrix(Reales, Predecidos, labels=list(range(10)))
print("\n", MatrizConfusion10Class)

# Matriz de Confusión de 2 Clases para cada número del 1 al 9
print("\n===== MATRICES DE CONFUSIÓN DE 2 CLASES (CADA NÚMERO vs OTROS) =====\n")

for numero in range(1, 10):
    # Convertir a 2 clases: 1 si es el número, 0 si es otro
    Reales_2Clases = [1 if x == numero else 0 for x in Reales]
    Predecidos_2Clases = [1 if x == numero else 0 for x in Predecidos]
    
    MatrizConfusion2Class = confusion_matrix(Reales_2Clases, Predecidos_2Clases, labels=[0, 1])
    
    print(f"NÚMERO {numero}:")
    print(f"Clase 0: No es {numero} | Clase 1: Es {numero}")
    print("Matriz de Confusión:")
    print(MatrizConfusion2Class)
    print(f"TN: {MatrizConfusion2Class[0, 0]} | FP: {MatrizConfusion2Class[0, 1]}")
    print(f"FN: {MatrizConfusion2Class[1, 0]} | TP: {MatrizConfusion2Class[1, 1]}")
    
    # Calcular métricas
    precision = precision_score(Reales_2Clases, Predecidos_2Clases, zero_division=0)
    recall = recall_score(Reales_2Clases, Predecidos_2Clases, zero_division=0)
    f1 = f1_score(Reales_2Clases, Predecidos_2Clases, zero_division=0)
    accuracy = accuracy_score(Reales_2Clases, Predecidos_2Clases)
    
    print(f"\nMétricas:")
    print(f"Precisión: {precision:.4f}")
    print(f"Recall (Sensibilidad): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Exactitud: {accuracy:.4f}")
    print()






