import tensorflow as tf
import tensorflow.keras as k
import matplotlib.pyplot as plt
from keras.models import Sequential
from PIL import Image
from matplotlib import image
from numpy import asarray
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.layers import Conv3D
from keras.layers import MaxPool2D

datagen = ImageDataGenerator(rescale=1/50,horizontal_flip=True,fill_mode='nearest')
train = datagen.flow_from_directory("seg_train/",class_mode="categorical", target_size=(32, 32),batch_size=64)
test = datagen.flow_from_directory("seg_test/",class_mode="categorical", target_size=(32, 32), batch_size=64)
validation_data = datagen.flow_from_directory("seg_pred/",class_mode="categorical", target_size=(32, 32), batch_size=64)

model = Sequential()
model.add(Conv2D(32,3,3,input_shape=(32,32,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(32,3,3,input_shape=(32,32,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(1024,activation='relu')) 
model.add(Dense(150,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(6,activation='softmax')) 
model.compile(k.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=\
    ["categorical_accuracy"])
model.fit_generator(train, epochs=7, validation_data=validation_data)
final = model.evaluate(test)











