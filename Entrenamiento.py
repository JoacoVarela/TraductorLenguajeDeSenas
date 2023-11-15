
#---------------------------------Importamos las fotos tomadas-----------------------------
import tensorflow as tf

#------------------------------ Crear modelo y entrenarlo ---------------------------------------
from tensorflow.keras.preprocessing.image import ImageDataGenerator  #Nos ayuda a preprocesar las imagenes que le entreguemos al modelo
from tensorflow.python.keras.models import Sequential  #Nos permite hacer redes neuronales secuenciales
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation #
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D  #Capas para hacer las convoluciones
from tensorflow.python.keras import backend as K        #Si hay una sesion de keras, lo cerramos para tener todo limpio
# from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.optimizer_v2.adam import Adam



datos_entrenamiento = 'C:/Users/Joaquin Varela/Documents/TraductorDeImagenes/Deteccion-y-Clasificacion-de-Manos/Fotos/Entrenamiento'
datos_validacion = 'C:/Users/Joaquin Varela/Documents/TraductorDeImagenes/Deteccion-y-Clasificacion-de-Manos/Fotos/Validacion'



#Parametros
iteraciones = 25  #Numero de iteraciones para ajustar nuestro modelo
altura, longitud = 200, 200 #Tamaño de las imagenes de entrenamiento
batch_size = 1  #Numero de imagenes que vamos a enviar
pasos = 350/1  #Numero de veces que se va a procesar la informacion en cada iteracion
pasos_validacion = 350/1 #Despues de cada iteracion, validamos lo anterior
filtrosconv1 = 32
filtrosconv2 = 64     #Numero de filtros que vamos a aplicar en cada convolucion
filtrosconv3 = 128     #Numero de filtros que vamos a aplicar en cada convolucion
tam_filtro1 = (4,4)
tam_filtro2 = (3,3)
tam_filtro3 = (2,2)   #Tamaños de los filtros 1 y 2
tam_pool = (2,2)  #Tamaño del filtro en max pooling
clases = 2  #Mano abierta y cerrada (5 dedos y 0 dedos)
lr = 0.0005  #ajustes de la red neuronal para acercarse a una solucion optima

#Pre-Procesamiento de las imagenes
preprocesamiento_entre = ImageDataGenerator(
    rescale= 1./255,   #Pasar los pixeles de 0 a 255 | 0 a 1
    shear_range = 0.3, #Generar nuestras imagenes inclinadas para un  mejor entrenamiento
    zoom_range = 0.3,  #Genera imagenes con zoom para un mejor entrenamiento
    horizontal_flip=True #Invierte las imagenes para mejor entrenamiento
)

preprocesamiento_vali = ImageDataGenerator(
    rescale = 1./255
)

imagen_entreno = preprocesamiento_entre.flow_from_directory(
    datos_entrenamiento,       #Va a tomar las fotos que ya almacenamos
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical',  #Clasificacion categorica = por clases
)

imagen_validacion = preprocesamiento_vali.flow_from_directory(
    datos_validacion,
    target_size=(altura,longitud),
    batch_size= batch_size,
    class_mode='categorical'
)

#Creamos la red neuronal convolucional (CNN)
cnn = Sequential()  #Red neuronal secuencial
#Agregamos filtros con el fin de volver nuestra imagen muy profunda pero pequeña
cnn.add(Convolution2D(filtrosconv1, tam_filtro1, padding = 'same', input_shape=(altura,longitud,3), activation = 'relu')) #Agregamos la primera capa
         #Es una convolucion y realizamos config
cnn.add(MaxPooling2D(pool_size=tam_pool)) #Despues de la primera capa vamos a tener una capa de max pooling y asignamos el tamaño

cnn.add(Convolution2D(filtrosconv2, tam_filtro2, padding = 'same', activation='relu')) #Agregamos nueva capa
cnn.add(MaxPooling2D(pool_size=tam_pool))
# nuevaCapa
cnn.add(Convolution2D(filtrosconv3, tam_filtro2, padding = 'same', activation='relu')) #Agregamos nueva capa
cnn.add(MaxPooling2D(pool_size=tam_pool))

#Ahora vamos a convertir esa imagen profunda a una plana, para tener 1 dimension con toda la info
cnn.add(Flatten())  #Aplanamos la imagen
cnn.add(Dense(256,activation='relu'))  #Asignamos 256 neuronas
cnn.add(Dropout(0.5)) #Apagamos el 50% de las neuronas en la funcion anterior para no sobreajustar la red
cnn.add(Dense(clases, activation='softmax'))  #Es nuestra ultima capa, es la que nos dice la probabilidad de que sea alguna de las clases

#Agregamos parametros para optimizar el modelo
#Durante el entrenamiento tenga una autoevalucion, que se optimice con Adam, y la metrica sera accuracy
# optimizar  = Adam(learning_rate=lr)
cnn.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# #Entrenaremos nuestra red
cnn.fit(imagen_entreno, steps_per_epoch=pasos, epochs= iteraciones, validation_data= imagen_validacion, validation_steps=pasos_validacion)

#Guardamos el modelo
cnn.save('ModeloVocales.h5')
cnn.save_weights('pesosVocales.h5', save_format='h5')

# Carga el modelo desde el archivo h5
modelo = tf.keras.models.load_model('ModeloVocales.h5')
K.clear_session()  #Limpiamos todo


# Carga los pesos del modelo
modelo.load_weights('pesosVocales.h5')

# Convierte el modelo a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
tflite_model = converter.convert()

with open('modelo_vocales.tflite', 'wb') as f:
    f.write(tflite_model)


