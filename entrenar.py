import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clear_session()

##Directorio donde tenemos las imagenes para entrenar y validar
data_entrenamiento='./data/entrenamiento'
data_validacion='./data/validacion'

##parametros

epocas=20
altura, longitud= 100, 100
batch_size=32
pasos=1000
pasos_validacion=200
filtrosConv1=32
filtrosConv2=64
tamano_filtro1=(3,3)
tamano_filtro2=(2,2)
tamano_pool=(2,2)
clases=3
lr=0.0005

##procesamiento de imagenes para entrenar

entrenamiento_datagen= ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

validacion_datagen= ImageDataGenerator(
    rescale=1./255
)

##entrara al directorio para procesar cada imagen
imagen_entrenamiento= entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

imagen_validacion=validacion_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

##Crear la red convolucional

cnn=Sequential()

cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding='same', input_shape=(altura, longitud, 3), Activation='relu'))

cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding='same', Activation='relu'))

cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())

cnn.add(Dense(256, Activation='relu'))

cnn.add(Dropout(0.5))

cnn.add(Dense(clases, Activation='softmax'))

cnn.conpile(less='categorical_crossentropy', optimizer=optimizers.adam(lr=lr), metrics=['accuracy'])

cnn.fit_generator(imagen_entrenamiento,steps_per_epoch=pasos, epochs=epocas, 
        validation_data=imagen_validacion, validation_steps=pasos_validacion)

##
dir='./modelo/'

##
if not os.path.exists(dir):
  os.mkdir(dir)
cnn.save('./modelo/modelo.h5')
cnn.save.weights('./modelo/pesos.h5')