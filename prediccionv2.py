# Lectura de la camara
import cv2
import SeguimientoManos as sm
from keras.models import load_model
import mediapipe as mp
import os
import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from keras.models import load_model
import flet as ft




modelo = "C:/Users/Joaquin Varela/Documents/TraductorDeImagenes/TraductorLenguajeDeSenas/ModeloVocales.h5"
peso =  'C:/Users/Joaquin Varela/Documents/TraductorDeImagenes/TraductorLenguajeDeSenas/pesosVocales.h5'
cnn = load_model(modelo)  #Cargamos el modelo
cnn.load_weights(peso)  #Cargamos los pesos

direccion = 'C:/Users/Joaquin Varela/Documents/TraductorDeImagenes/Deteccion-y-Clasificacion-de-Manos/Fotos/Validacion'
dire_img = os.listdir(direccion)
print("Nombres: ", dire_img)

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector= sm.detectormanos(maxManos=1,Confdeteccion=0.9)

while True:
    ret,frame = cap.read()

    mano = detector.encontrarmanos(frame)

    lista,bbox,manoFake = detector.encontrarposicion(frame)

    if len(lista) != 0:
        xmin,ymin, xmax,ymax =bbox

        xmin=xmin-40
        ymin=ymin-40
        xmax=xmax+40
        ymax=ymax+40

        data = frame[ymin:ymax, xmin:xmax]

        obj = cv2.resize(data, (200, 200), interpolation=cv2.INTER_CUBIC)
       

        x = img_to_array(obj)  # Convertimos la imagen a una matriz
        x = np.expand_dims(x, axis=0)  # Agregamos nuevo eje
        vector = cnn.predict(x)  # Va a ser un arreglo de 2 dimensiones, donde va a poner 1 en la clase que crea correcta
        resultado = vector[0]  # [1,0] | [0, 1]
        respuesta = np.argmax(resultado)  
        
        if respuesta == 0:
            print(resultado)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, '{}'.format(dire_img[0]), (xmin, ymin - 5), 1, 1.3, (0, 255, 0), 1, cv2.LINE_AA)
        elif respuesta == 1:
            print(resultado)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, '{}'.format(dire_img[1]), (xmin, ymin - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
        elif respuesta == 2:
            print(resultado)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, '{}'.format(dire_img[2]), (xmin, ymin - 5), 1, 1.3, (0, 255, 255), 1, cv2.LINE_AA)
        else: 
            cv2.putText(frame, 'Letra Desconocida', (xmin, ymin - 5), 1, 1.3, (0, 255, 255), 1, cv2.LINE_AA)


    cv2.imshow('LENGUAJE DE VOCALES', frame)
    t=cv2.waitKey(1)
    if t==27:
        break
        
cap.release()
cv2.destroyAllWindows()
