from Model import count_weights
from math import factorial
import numpy as np

def complex(net):
    complexity = 1
    for i in range(1, len(net)-1):
        complexity *= (net[i] // net[0]) ** net[0]
    aux = 0
    for j in range(net[0]):
        aux += factorial(len(net) - 1) / factorial(len(net) - j)
    complexity *= aux

    return complexity

def calculate_train_size(data, qnt_data):
    # rows = qnt_data/data.shape[1]-1
    # pct = (rows*100)/data.shape[0]
    # pct /= 100
    pct = qnt_data/len(data)

    return round(pct, 6)

def normalization(train, test):
    features = (train.shape[1])

    for i in range(features):
        min_t = train.iloc[:, i].min()
        max_t = train.iloc[:, i].max()

        train.iloc[:, i] = (train.iloc[:, i]-min_t)/(max_t-min_t)
        test.iloc[:, i] = (test.iloc[:, i]-min_t)/(max_t-min_t)

    return train, test

def write_file(path, experiment, array, string=''):
    array = np.array(array)
    data_size = experiment.get_train_size()
    
    if string == 'loss_val':
        path += 'loss_val.csv'

    elif string == 'loss_train':
        path += 'loss_train.csv'

    elif string == 'loss_test':
        path += 'loss_test.csv'

    elif string == 'acc_train':
        path += 'acc_train.csv'
    
    elif string == 'acc_val':
        path += 'acc_val.csv'

    else:
        path += 'acc_test.csv'

    with open(path, 'a') as file:
        file.write(f'ARQ{experiment.model.neurons},')
        file.write(f'{experiment.model.neurons},')
        file.write(f'{data_size},')
        file.write(f'{count_weights(experiment.model)},')
        # array = *1, *2, *3, *4, *5 
        # *_M, *_DP
        for i in range(5):
            file.write(f'{array[i]},')
        file.write(f'{array.sum()/len(array)},')
        file.write(f'{np.std(array)}')
        file.write('\n')
        file.close()

    return 


