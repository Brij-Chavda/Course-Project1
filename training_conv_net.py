# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1W46AiLAR45AnN9kjq4LCFIzU9rhMygGW
"""

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

def create_LeNet():
    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=3, padding='same', input_shape=(28, 28,1), activation='relu'),
        keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2), 
        keras.layers.Flatten(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

from keras.preprocessing.image import ImageDataGenerator
 datagen = ImageDataGenerator(horizontal_flip = 1)
 confusion_m = []
 loss_m = []
 if __name__ == "__main__":

    number_epochs = 30

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255
    
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    model = create_LeNet()
    network = model.fit_generator(datagen.flow(train_images,train_labels,batch_size=300),steps_per_epoch=200,epochs=number_epochs)

test_loss, test_acc = model.evaluate(test_images,test_labels)
test_acc

model.save('models/convnet_model.h5')

loss_m = network.history["loss"]
plt.plot(loss_m)
plt.xlabel('no of epochs') 
plt.ylabel('loss')  
plt.title('ConvNet loss')

predicted_data=(np.argmax(model.predict(test_images),1))

from sklearn.metrics import confusion_matrix
confusion_matrix(test_labels , predicted_data )