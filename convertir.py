import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib import lite

'''
converter = lite.TFLiteConverter.from_keras_model_file('/home/andres/Documentos/TFG/modelo/modelo.h5')
tfmodel = converter.convert()
#open ("model.tflite" , "wb").write(tfmodel)
'''

'''
# Converting a SavedModel.
converter = lite.TFLiteConverter.from_saved_model('/home/andres/Documentos/TFG/modelo/modelo.h5')
tflite_model = converter.convert()'''


#Convert a tf.Keras model 
#The following example converts a tf.keras model into a TensorFlow Lite Flatbuffer. The tf.keras file must contain both the model and the weights.

#tflite_convert --output_file=/tmp/foo.tflite --keras_model_file=/tmp/keras_model.h5
'''
#video yt
#funciona--> 
#converter = lite.TocoConverter.from_keras_model_file('/home/andres/Documentos/TFG/modelo/modelo.h5')
converter = lite.TFLiteConverter.from_keras_model_file('/home/andres/Documentos/TFG/modelo/modelo.h5')
tfmodel = converter.convert()
open("/home/andres/Documentos/TFG/model.tflite","wb").write(tfmodel)
'''
'''
from tensorflow.python.keras import applications
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation, GlobalAveragePooling2D

clases=3
pesos_modelo = '/home/andres/Documentos/TFG/modelo/probabilidades1.h5'
vgg=applications.vgg16.VGG16()
converter=Sequential()
for capa in vgg.layers:
    converter.add(capa)
converter.pop()
for layer in converter.layers:
    layer.trainable=False
converter.add(Dense(clases,activation='softmax'))  
converter.load_weights(pesos_modelo)  # Cargamos todo el Aprendizaje ya hecho por nuestra red neuronal en variable cnn
# a partir de aqui , def Predict (file) ............

'''




'''
pretrained_model=tf.keras.applications.MobileNet
tf.saved_model.save(pretrained_model, "/tmp/mobilenet/1/")
tf.saved_model.simple_save(modelo, '/home/andres/Documentos/TFG/saveModel/')
'''

'''
#quantization
converter = lite.TFLiteConverter.from_saved_model(modelo)
converter.post_training_quantize = True #
converter.optimizations = [lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
open("/home/andres/Documentos/TFG/modelQuantization.tflite","wb").write(tflite_quant_model)
'''


modelo='/home/andres/Documentos/TFG/modelo/modelo.h5'
modelo1='/home/andres/Documentos/TFG/modelo/modelo1.h5'

#okkkkkkkkkkkkkkkkkkkk
converter = lite.TFLiteConverter.from_keras_model_file(modelo)
#converter = lite.TocoConverter.from_keras_model_file(modelo1)
print("a")
tfmodel = converter.convert()
open("/home/andres/Documentos/TFG/tflite/model.tflite","wb").write(tfmodel)
