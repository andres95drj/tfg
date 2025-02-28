# coding=utf-8

import argparse
import os
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as preprocess_input19
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_input16
from keras.utils import np_utils
import numpy as np

from numpy.random import seed
seed(31416)
from tensorflow import set_random_seed
set_random_seed(31416)

ARGS = None

# Función que realiza el transfer learning y devuelve el accuracy obtenido
def transferLearning(image_dir, imgs_rows, imgs_cols, batch_size, epochs,
                        modelT):
    channel = 3

    classes = next(os.walk(image_dir + '/entrenamiento'))[1]

    if modelT == 'vgg16':
        datagen = ImageDataGenerator(preprocessing_function = preprocess_input16)
        datagen_test = ImageDataGenerator(preprocessing_function = preprocess_input16)
    else:
        datagen = ImageDataGenerator(preprocessing_function = preprocess_input19)
        datagen_test = ImageDataGenerator(preprocessing_function = preprocess_input19)

    train_generator = datagen.flow_from_directory(image_dir + '/entrenamiento',#train
                        target_size = (imgs_cols, imgs_rows),
                        batch_size = batch_size, class_mode = 'categorical',
                        shuffle = False)

    test_generator = datagen_test.flow_from_directory(image_dir + '/validacion',
                        target_size = (imgs_cols, imgs_rows),
                        batch_size = batch_size, class_mode = 'categorical',
                        shuffle = False)

    # Load our model
    if modelT == 'vgg16':
        model = VGG16(include_top = False, weights='imagenet', pooling = 'max')
    else:
        model = VGG19(weights = 'imagenet', include_top = False, pooling = 'max')

    
    for layer in model.layers[:15]:#15
        layer.trainable = False

    features = model.predict_generator(train_generator, steps=10)
    train_labels = np_utils.to_categorical(train_generator.classes, len(classes))

    test_features = model.predict_generator(test_generator, steps=10)
    test_labels = np_utils.to_categorical(test_generator.classes, len(classes))

    # Creamos un nuevo modelo, de forma que entrenemos sólo la última capa
    new_model = Sequential()
    #new_model.add(Flatten()) # O bien no se usa Flatten y se usa el arguemnto 'pooling = 'max'' al llamar a VGG o bien no se usa y se usa Flatten aquí
    new_model.add(Dense(512, activation = 'relu', input_shape = features.shape[1:])) # Si se usa Flatten hay que quitar el argumento input_shape aquí
    new_model.add(Dense(len(classes), activation = 'softmax'))

    opt = SGD(lr = 1e-3, decay = 1e-6, momentum = 0.9, nesterov = True)
    new_model.compile(optimizer = opt, loss = 'categorical_crossentropy',
                            metrics = ['accuracy'])
    new_model.fit(features, train_labels, epochs = epochs,
                        batch_size = batch_size)

    # Make predictions
    predictions = new_model.predict(test_features, batch_size = batch_size,
                                            verbose = 1)

    # Calculate accuracy
    test_labels = np.argmax(test_labels, axis = 1)
    predictions = np.argmax(predictions, axis = 1)
    acc = sum(test_labels == predictions)/len(test_labels)

    return acc

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type = str,
        default = '',
        help = "Path to folders of labeled images."
    )
    parser.add_argument(
        '--imgs_rows',
        type = int,
        default = 224,
        help = "Number of rows in images, after reshape"
    )
    parser.add_argument(
        '--imgs_cols',
        type = int,
        default = 224,
        help = "Number of cols in images, after reshape"
    )
    parser.add_argument(
        '--batch_size',
        type = int,
        default = 8,
        help = "Batch size"
    )
    parser.add_argument(
        '--epochs',
        type = int,
        default = 10,
        help = "Number of epochs to train"
    )
    parser.add_argument(
        '--model',
        type = str,
        default = 'vgg16',
        help = "Model to use. It must be 'vgg16' or 'vgg19'."
    )
    ARGS, unparsed = parser.parse_known_args()

    if ARGS.model != 'vgg16' and ARGS.model != 'vgg19':
        print("model must be 'vgg16' or 'vgg19'.")
        sys.exit(1)

    acc = transferLearning(ARGS.image_dir, ARGS.imgs_rows, ARGS.imgs_cols,
                            ARGS.batch_size, ARGS.epochs, ARGS.model)
    print(acc)
