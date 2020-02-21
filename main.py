import numpy as np
import pdb
import os
import collections, numpy
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

def test(model, test_images, test_labels):    
    loss, acc = model.evaluate(test_images, test_labels)
    ypred = model.predict(test_images)
    ypred = np.argmax(ypred, axis=1)
    return loss, test_labels, ypred


if __name__ == "__main__":
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    
    mlp_model = tf.keras.models.load_model('models/MLP_model.h5')

    convnet_model = tf.keras.models.load_model('models/convnet_model.h5')
    
    loss, get_label, pred = test(mlp_model, test_images, test_labels)
    with open("multi-layer-net.txt", 'w') as file:
        file.write("Loss on Test Data : {}\n".format(loss))
        file.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(get_label) == np.array(pred))))
        file.write("actual_label,pred_label \n")
        for index in range(len(get_label)):
            file.write("{}\t{}\n".format(get_label[index], pred[index]))

    loss, get_label, pred = test(convnet_model, test_images, test_labels)
    with open("convolution-neural-net.txt", 'w') as file:
        file.write("Loss on Test Data : {}\n".format(loss))
        file.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(get_label) == np.array(pred))))
        file.write("actual_label,pred_label \n")
        for index in range(len(get_label)):
            file.write("{}\t{}\n".format(get_label[index], pred[index]))