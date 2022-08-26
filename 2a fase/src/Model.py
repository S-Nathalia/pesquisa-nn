import tensorflow.keras.backend as K
import tensorflow as tf
import pandas as pd
import numpy as np

class Model(tf.keras.Model):
    def __init__(self, data, neurons):
        super(Model, self).__init__()
        self.layers_model = tf.keras.Sequential()
        self.shape = data.shape[1]-1
        self.num_layers = 0
        self.neurons = neurons
        self.input_layer = tf.keras.layers.Input(shape=(self.shape,))
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

        self.neurons_layers = self.__calculate_layers()
        self.__init_layers()

    def __calculate_layers(self):
        self.num_layers = self.neurons//self.shape
        neurons_layers = np.ones(self.num_layers, dtype='uint8')*self.shape

        if(self.neurons%self.shape != 0):
            self.num_layers += 1
            neurons_layers = np.insert(neurons_layers, len(neurons_layers), self.neurons%self.shape)

        return neurons_layers


    def __init_layers(self):
        for lr in range(self.num_layers):
            self.layers_model.add(tf.keras.layers.Dense(self.neurons_layers[lr], activation="relu"))

    def __call__(self, inputs, training=False):
        x = self.layers_model(inputs)
        x = self.output_layer(x)
        return x

def count_weights(model):
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])

    return trainable_count