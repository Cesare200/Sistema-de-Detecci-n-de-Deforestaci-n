import sys
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

K.clear_session()

data_entrenamiento = 'C:\\Users\\HP\\Desktop\\ML4\\ML-bosques\\data\\entrenamiento'
data_validacion = 'C:\\Users\\HP\\Desktop\\ML4\\ML-bosques\\data\\validacion'



# Parámetros
epocas = 10
altura, longitud = 100, 100
batch_size = 1
pasos = 83
pasos_validacion = 20
filtrosConv1 = 32
filtrosConv2 = 64
tamaño_filtro1 = (3, 3)
tamaño_filtro2 = (2, 2)
tamaño_pool = (2, 2)
clases = 2  # Estas son las etiquetas perro, gato, gorila
lr = 0.005  # Tasa de aprendizaje

# Preprocesamiento de imágenes
entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

validacion_datagen = ImageDataGenerator(
    rescale=1. / 255
)

imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

imagen_validacion = validacion_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

# Crear red neuronal convolucional
cnn = Sequential()

cnn.add(Conv2D(filtrosConv1, tamaño_filtro1, padding='same', input_shape=(altura, longitud, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamaño_pool))

cnn.add(Conv2D(filtrosConv2, tamaño_filtro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamaño_pool))

cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation='softmax'))  # Salida

cnn.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=lr), metrics=['accuracy'])

cnn.fit(imagen_entrenamiento, steps_per_epoch=pasos, epochs=epocas, validation_data=imagen_validacion, validation_steps=pasos_validacion)


target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')

