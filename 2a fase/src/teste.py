import numpy as np
import pandas as pd

def calculate_layers():
    num_layers = neurons//shape
    neurons_layers = np.ones(num_layers, dtype='uint8')*shape

    if(neurons%shape != 0):
        num_layers += 1
        neurons_layers = np.insert(neurons_layers, len(neurons_layers), neurons%shape)

    print(num_layers, neurons_layers, neurons_layers.sum())

shape = 12
neurons = 15

calculate_layers()