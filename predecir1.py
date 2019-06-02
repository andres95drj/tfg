import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3

import tensorflow as tf
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.vgg16 import VGG16
import keras
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.models import load_model, Model


from tensorflow.python.keras import applications
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation, GlobalAveragePooling2D



modelo = '/home/andres/Documentos/TFG/modelo/modelo1.h5'
pesos_modelo = '/home/andres/Documentos/TFG/pesos/probabilidades1.h5'

longitud, altura = 224, 224
clases=3

#with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
#        cnn = load_model(modelo)

#cnn = load_model(modelo) #cargamos los archivos que generamos para poder reutilizar nuestra red neuronal
#cnn.load_weights(pesos_modelo)


#model = InceptionV3(weights='imagenet', include_top=True)


#original
vgg=applications.vgg16.VGG16()
cnn=Sequential()
for capa in vgg.layers:
    cnn.add(capa)
cnn.pop()
for layer in cnn.layers:
    layer.trainable=False
cnn.add(Dense(clases,activation='softmax'))  
cnn.load_weights(pesos_modelo)  # Cargamos todo el Aprendizaje ya hecho por nuestra red neuronal en variable cnn
# a partir de aqui , def Predict (file) ............





'''
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        cnn = load_model(modelo)


cnn.load_weights(pesos_modelo)
'''


import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
#from keras.applications.vgg16 import preprocess_input, decode_predictions
'''
#nuevo1
def predict(file):
  x = load_img(file, target_size=(longitud, altura))#cargamos la imagen que queremos que nos prediga
  x = img_to_array(x)#pasamos imagen a vector
  x = np.expand_dims(x, axis=0)#anadimos una dimension extra en nuestra 1era dimension para procesar nuestra informacion sin problema
  array = cnn.predict(x) #[[1,0,0,0,0]]
  
  #ejemplo=cnn.predict_on_batch(x)
  #print("\nEJEMPLO: ",ejemplo,"\n")
  result = array[0] #[0,0,1]
  #tf.nn.softmax(5)
  #test_prediction = tf.nn.softmax(array)
  #print("Precision en el conjunto de PRUEBA: %.1f%%" % precision(test_prediction.eval(session = sess), '/home/andres/Documentos/TFG/extintor.jpeg'))
  
  #print("\nARRAY: ",array[0],"\n")
  answer = np.argmax(result)#indice del valor mas alto. #2
  if answer == 0:
    print("pred: extintores")
  elif answer == 1:
    print("pred: lanzas")
  elif answer == 2:
    print("pred: motosierras")
  """elif answer == 3:
    print("pred: motosierras")
  elif answer == 4:
    print("pred: reduccion")"""

  return answer
'''


#nuevo2
#FUNCIONA

def predict(file):
  x = load_img(file, target_size=(longitud, altura))#cargamos la imagen que queremos que nos prediga
  x = img_to_array(x)#pasamos imagen a vector
  x = np.expand_dims(x, axis=0)#anadimos una dimension extra en nuestra 1era dimension para procesar nuestra informacion sin problema
  array = cnn.predict(x) #[[1,0,0,0,0]]
  
  #ejemplo=cnn.predict_on_batch(x)
  #print("\nEJEMPLO: ",ejemplo,"\n")
  result = array[0] #[0,0,1]
  #tf.nn.softmax(5)
  #test_prediction = tf.nn.softmax(array)
  #print("Precision en el conjunto de PRUEBA: %.1f%%" % precision(test_prediction.eval(session = sess), '/home/andres/Documentos/TFG/extintor.jpeg'))
  
  #print("\nARRAY: ",array[0],"\n")
  answer = np.argmax(result)#indice del valor mas alto. #2
  if answer == 0:
    print("pred: extintores")
  elif answer == 1:
    print("pred: lanzas")
  elif answer == 2:
    print("pred: motosierras")
  """elif answer == 3:
    print("pred: motosierras")
  elif answer == 4:
    print("pred: reduccion")"""

  return answer
  #funciona 




'''original
def predict(model, img_path, target_size=(224, 224), top_n=3):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return decode_predictions(preds, top=top_n)[0]


import matplotlib.pyplot as plt


def plot_image(img_path):
    img = image.load_img(fn, target_size=(299, 299))
    plt.figure(figsize=(8, 8))
    plt.imshow(img, interpolation='nearest')
    plt.axis('off')
    
def plot_pred(pred):
    plt.figure(figsize=(8, 2))
    classes = [c[1] for c in pred]
    probas = [c[2] for c in pred]
    y_pos = np.arange(len(classes))
    plt.barh(y_pos, probas, align='center')
    plt.yticks(y_pos, classes)
    plt.gca().invert_yaxis()
    plt.xlabel('Probability')
    plt.xlim(0, 1)

'''

fn='/home/andres/Documentos/TFG/motosierra2.jpeg'
predict(fn)


#fn='/home/andres/Escritorio/pruebasPython/ejemplos/tigre.jpeg'

#pred = predict(cnn, fn)
#print(pred)

#print("a\n")
#plot_image(fn) #
#plot_pred(pred) #
#print("A")