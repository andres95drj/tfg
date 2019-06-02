import sys
import os
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.python.keras import backend as K
import numpy as np
import keras
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras import applications

K.clear_session()

data_entrenamiento = '/home/andres/Documentos/TFG/imagenes/entrenamiento'
data_validacion = '/home/andres/Documentos/TFG/imagenes/validacion'

"""
Parameters
"""
epocas = 5 #20
longitud, altura = 224, 224
#longitud, altura = 150, 150
batch_size = 4 #32
pasos = 10 #1000
validation_steps = 5 #300
filtrosConv1 = 3 #32 ########este cambia la primera capa
filtrosConv2 = 16 #64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 3
lr = 0.0005 #Learning rate, ajuste que realiza


# Preparamos nuestras imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255, #pixel en vez de 1-255 van de 0-1 (mas eficiente entrenamiento)
    shear_range=0.2, #inclinar
    zoom_range=0.2, #aumenta algunas imagenes
    horizontal_flip=True) #invertir

test_datagen = ImageDataGenerator(rescale=1. / 255) #imagenes tal cual para validar

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento, #abre todas las carpetas 
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')#clasificacion categorica

validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')


model = applications.VGG16(weights='imagenet', include_top=False)

#cnn = applications.VGG16(weights='imagenet') #correo
cnn = Sequential()
#cnn.add(GlobalMaxPooling2D(input_shape=model.output_shape[1:])) #flatten input_shape=model.output_shape[1:]
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding="same", input_shape=(longitud, altura, 3), activation='relu')) #1era capa inputshape solo en 1era capa
cnn.add(Flatten()) #add(Flatten(input_shape=model.output_shape[1:]))
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(3, activation='softmax'))



'''
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding="same", input_shape=(longitud, altura, 3), activation='relu')) #1era capa inputshape solo en 1era capa
cnn.add(MaxPooling2D(pool_size=tamano_pool))#1era capa de pooling 
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding="same"))#2da capa convolucional activation='relu'
cnn.add(MaxPooling2D(pool_size=tamano_pool)) 
cnn.add(Flatten())#hacemos plana la imagen --> 1D 
cnn.add(Dense(256, activation='relu')) #256 neuronas/ mandamos a una capa normal interconexion d neuronas
cnn.add(Dropout(0.5)) #durante el entreno apagamos 50% de las 256neuronas para evitar el sobreentramiento. asi genera un modelo que se adpta mejor a nuevos datos
cnn.add(Dense(clases, activation='softmax'))#ultima capa 5neuronas predice


cnn.compile(loss='categorical_crossentropy', #funcion de perdida
            optimizer=optimizers.Adam(lr=lr),#optimizador con learning rate como indicamos arriba
            metrics=['accuracy'])#porcentaje de que tan bien esta aprendiendo nuestra red
'''
cnn.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

cnn.fit_generator( #entrenamos red neuronal con
    entrenamiento_generador,#imagenes d entrenamiento
    steps_per_epoch=pasos,#cada epoca 1000 pasos  EPOCA1: 1000 pasos-->300 pasos de validacion--> pasa a EPOCA2 1000 pasos-->300 pasos de validacion--> pasa a EPOCA3 ...
    epochs=epocas,#20 epcoas
    validation_data=validacion_generador,##callbacks = [cp_callback], #nuevo!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    validation_steps=validation_steps)#300

target_dir = '/home/andres/Documentos/TFG/modelo/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
cnn.save('/home/andres/Documentos/TFG/modelo/top_model.h5')

target_dir1 = '/home/andres/Documentos/TFG/pesos/'
if not os.path.exists(target_dir1):
    os.mkdir(target_dir1)
cnn.save_weights('/home/andres/Documentos/TFG/pesos/weights_top_model.h5')

target_dir2 = '/home/andres/Documentos/TFG/labels/'
if not os.path.exists(target_dir2):
    os.mkdir(target_dir2)
f = open ("/home/andres/Documentos/TFG/labels/labelsT.txt", "w")
for a in entrenamiento_generador.class_indices:
    x=a+"\n"
    f.write(x)    

f.close()


'''
target_dir = '/home/andres/Documentos/TFG/modelo/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
cnn.save('/home/andres/Documentos/TFG/modelo/top_model.h5')
cnn.save_weights('/home/andres/Documentos/TFG/modelo/weights_top_model.h5')
'''

cnn.summary()
