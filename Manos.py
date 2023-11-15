import cv2
import os
import SeguimientoManos as sm

nombre = 'Letra_C'
direccion = 'C:/Users/Joaquin Varela/Documents/TraductorDeImagenes/Deteccion-y-Clasificacion-de-Manos/Fotos/Validacion'
carpeta = direccion + '/' + nombre

if not os.path.exists(carpeta):
    print('carpeta creada:', carpeta)
    os.mkdir(carpeta)

# Lectura de la camara
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

cont = 0

detector= sm.detectormanos(Confdeteccion=0.9)

while True:
    ret,frame = cap.read()

    frame = detector.encontrarmanos(frame, dibujar=False)

    lista1,bbox,mano = detector.encontrarposicion(frame,ManoNum=0, dibujarPuntos= False, dibujarBox= False, color=[0,255,0])

    if mano == 1:
        xmin,ymin, xmax,ymax =bbox

        xmin=xmin-40
        ymin=ymin-40
        xmax=xmax+40
        ymax=ymax+40

        recorte = frame[ymin:ymax, xmin:xmax]

        recorte= cv2.resize(recorte, (200,200), interpolation =cv2.INTER_CUBIC)

        cv2.imwrite(carpeta + "/C_{}.jpg".format(cont), recorte)
        cont = cont + 1

        # cv2.rectangle(frame,(xmin, ymin ), (xmax, ymax ), [0,255,0],2)

    cv2.imshow('LENGUAJE DE VOCALES', frame)
    t=cv2.waitKey(1)
    if t==27 or cont==350:
        break
        
cap.release()
cv2.destroyAllWindows()

    