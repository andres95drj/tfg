from keras.applications.resnet50 import decode_predictions


import numpy as np
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input#, decode_predictions
from keras.models import load_model #, load_weights

import tensorflow as tf

from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

#longitud, altura = 150, 150
longitud, altura = 224, 224
modelo = '/home/andres/Documentos/TFG/modelo/modelo.h5'
pesos_modelo = '/home/andres/Documentos/TFG/pesos/probabilidades.h5'

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        cnn = load_model(modelo)

#cnn = load_model(modelo) #cargamos los archivos que generamos para poder reutilizar nuestra red neuronal
cnn.load_weights(pesos_modelo)


cnn.summary()


# OK
def predict(file):
  x = load_img(file, target_size=(longitud, altura))#cargamos la imagen que queremos que nos prediga
  x = img_to_array(x)#pasamos imagen a vector
  x = np.expand_dims(x, axis=0)#anadimos una dimension extra en nuestra 1era dimension para procesar nuestra informacion sin problema
  array = cnn.predict(x) #[[1,0,0,0,0]]
  result = array[0] #[0,0,1]
  
  answer = np.argmax(result)#indice del valor mas alto. #2
  if answer == 0:
    print("pred: extintores")
  elif answer == 1:
    print("pred: lanzas")
  elif answer == 2:
    print("pred: motosierras")

  return answer


p=predict('/home/andres/Documentos/TFG/lanza.jpeg')
print("\nEJEMPLO: ",p)