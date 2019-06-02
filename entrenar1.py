"""
fichero que crea una red neuronal usando la red VGG16 y modificando sus salidas y realizando fine-tunning
y guarda su modelo, pesos y etiquetas
"""

import sys
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
#from tensorflow.python.keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation, Convolution2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
#from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation, Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.python.keras import backend as K
#from tensorflow.python.keras import applications
from keras import applications
from keras.utils.vis_utils import plot_model
import keras


K.clear_session()

data_entrenamiento = '/home/andres/Documentos/TFG/imagenes/entrenamiento'
data_validacion = '/home/andres/Documentos/TFG/imagenes/validacion'

epocas = 10 
longitud, altura = 224, 224
batch_size = 16 #16 
pasos = 500 
validation_steps = 100 
clases = 3
lr = 0.0005 #Learning rate, ajuste que realiza

filtrosConv1 = 8 #32
filtrosConv2 = 16 #64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)

def modelo():
        
    #vgg=applications.vgg16.VGG16(weights='imagenet', classes=clases, include_top=False) #vgg16 con 3 salidas
    #vgg.summary()
    
    top_model_weights_path = '/home/andres/Documentos/TFG/pesos/weights_top_model.h5'

    cnn = applications.VGG16(weights='imagenet',include_top=False) #correo
    #plot_model(cnn, to_file='/home/andres/Documentos/TFG/imagenes/cnn.png')

    '''
    x = applications.VGG16(weights='imagenet',include_top=False) #correo

    last_layer = x.layers[-2].output ## 
    output = keras.layers.Dense(3, activation="softmax")(last_layer) 

    cnn = keras.models.Model(inputs=x.inputs, outputs=output)
    '''

    cnn.summary()
    
    top_model = Sequential()
    top_model.add(Flatten())
    #top_model.add(GlobalMaxPooling2D(input_shape=cnn.output_shape[1:])) #flatten input_shape=model.output_shape[1:]
    top_model.add(Dense(256, activation='relu'))#, input_shape=cnn.output_shape[1:]))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(3, activation='softmax'))
    
    top_model.load_weights(top_model_weights_path)

    cnn.add(top_model)

    for layer in cnn.layers[:15]:#15
        layer.trainable = False


    '''
    vgg=Sequential()
    vgg=applications.vgg16.VGG16()
    cnn=Sequential()
    
    #print("VGG\n")
    #vgg.summary()

    for capa in vgg.layers: 
       cnn.add(capa)

    print("ANTES\n")
    cnn.summary()
    
    cnn.pop()
    #cnn.layers.pop()
    
    print("POP\n")
    cnn.summary()

    for layer in cnn.layers:
        layer.trainable=False
    cnn.add(Dense(clases,activation='softmax',name='predictions'))
    
    print("PRED\n")
    cnn.summary()
    
    '''

    return cnn #model

# Preparamos las imagenes

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

cnn=modelo() #cargamos cnn

cnn.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
'''
cnn.compile(loss='categorical_crossentropy', #funcion de perdida
            optimizer=optimizers.Adam(lr=lr),#optimizador con learning rate como indicamos arriba
            metrics=['accuracy'])#porcentaje de que tan bien esta aprendiendo nuestra red
'''

cnn.fit_generator( #entrenamos cnn
    entrenamiento_generador,#imagenes d entrenamiento
    steps_per_epoch=pasos,#cada epoca 1000 pasos  EPOCA1: 1000 pasos-->300 pasos de validacion--> pasa a EPOCA2 1000 pasos-->300 pasos de validacion--> pasa a EPOCA3 ...
    epochs=epocas,#20 epcoas
    validation_data=validacion_generador,#
    validation_steps=validation_steps)#300


target_dir = '/home/andres/Documentos/TFG/modelo/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
cnn.save('/home/andres/Documentos/TFG/modelo/modelo1.h5')

target_dir1 = '/home/andres/Documentos/TFG/pesos/'
if not os.path.exists(target_dir1):
    os.mkdir(target_dir1)
cnn.save_weights('/home/andres/Documentos/TFG/pesos/probabilidades1.h5')

target_dir2 = '/home/andres/Documentos/TFG/labels/'
if not os.path.exists(target_dir2):
    os.mkdir(target_dir2)
f = open ("/home/andres/Documentos/TFG/labels/labels1.txt", "w")
for a in entrenamiento_generador.class_indices:
    x=a+"\n"
    f.write(x)    

f.close()