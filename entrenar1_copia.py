
import sys
import os
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import applications
from keras.utils.vis_utils import plot_model


K.clear_session()
#vgg=applications.vgg16.VGG16() 


#with tensorflow.device('/GPU:0'):



data_entrenamiento = '/home/andres/Documentos/TFG/imagenes/entrenamiento'
data_validacion = '/home/andres/Documentos/TFG/imagenes/validacion'

"""
Parameters
"""
epocas = 10 #20
longitud, altura = 224, 224
batch_size = 16 #32
pasos = 500 #1000
validation_steps = 100 #300
'''
filtrosConv1 = 8 #32
filtrosConv2 = 16 #64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
'''
clases = 3
lr = 0.0005 #Learning rate, ajuste que realiza

#v3=applications.inception_v3.InceptionV3()

def modelo():
    cnn=applications.vgg16.VGG16(weights=None, classes=clases)
    
    #plot_model(cnn, to_file='/home/andres/Documentos/TFG/imagenes/cnn.png')

    #for layer in cnn.layers:
        #config = layer.get_config()#{'name': 'predictions', 'trainable': True, 'dtype': 'float32', 'units': 3, 'activation': 'softmax', 'use_bias': True, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None, 'dtype': 'float32'}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {'dtype': 'float32'}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}
        
        #if(layer.name != "predictions"):
        #    layer.trainable=False

    #cnn.layers



    #vgg=applications.vgg16.VGG16()
    #vgg.summary()
    
    #cnn=Sequential()
    
    
    
    #cnn.add(InputLayer(3, tamano_filtro1, name='Input_layer', padding="same", input_shape=(longitud, altura, 3), activation='relu')) #1era capa inputshape solo en 1era capa
    #cnn.add(tf.keras.layers.Conv2D(3, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    #cnn.add(Convolution2D(3, tamano_filtro1, padding="same", input_shape=(longitud, altura, 3), activation='relu')) #1era capa inputshape solo en 1era capa
    cnn.summary()
    #cnn=vgg

    for layer in cnn.layers[:15]:
        layer.trainable = False

    '''
    cnn.add(vgg.layers[0])
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
    
    print("DESPUES\n")
    cnn.summary()
    '''
    return cnn




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

cnn=modelo()

#print("\nDESPUES\n")
#cnn.summary()##############

cnn.compile(loss='categorical_crossentropy', #funcion de perdida
            optimizer=optimizers.Adam(lr=lr),#optimizador con learning rate como indicamos arriba
            metrics=['accuracy'])#porcentaje de que tan bien esta aprendiendo nuestra red


cnn.fit_generator( #entrenamos red neuronal con
    entrenamiento_generador,#imagenes d entrenamiento
    steps_per_epoch=pasos,#cada epoca 1000 pasos  EPOCA1: 1000 pasos-->300 pasos de validacion--> pasa a EPOCA2 1000 pasos-->300 pasos de validacion--> pasa a EPOCA3 ...
    epochs=epocas,#20 epcoas
    validation_data=validacion_generador,#
    validation_steps=validation_steps)#300


target_dir = '/home/andres/Documentos/TFG/modelo/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
cnn.save('/home/andres/Documentos/TFG/modelo/modelo1c.h5')

target_dir1 = '/home/andres/Documentos/TFG/pesos/'
if not os.path.exists(target_dir1):
    os.mkdir(target_dir1)
cnn.save_weights('/home/andres/Documentos/TFG/pesos/probabilidades1c.h5')

target_dir2 = '/home/andres/Documentos/TFG/labels/'
if not os.path.exists(target_dir2):
    os.mkdir(target_dir2)

f = open ("/home/andres/Documentos/TFG/labels/labels1c.txt", "w")
for a in entrenamiento_generador.class_indices:
    x=a+"\n"
    f.write(x)    
f.close()

