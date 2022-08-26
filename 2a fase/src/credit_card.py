from sklearn.impute import SimpleImputer
from Experiment import *
import tensorflow as tf
import pandas as pd
import numpy as np
from Model import *


def write_file(path, data_size, exp, neurons, model, array):
    array = np.array(array)
    with open(path, 'a') as file:

        file.write(f'{exp},')
        file.write(f'{data_size},')
        file.write(f'{model.neurons},')
        file.write(f'{count_weights(model)},')
        for i in range(6):
            file.write(f'{array[i]},')
        file.write('\n')
        file.close()

data = pd.read_csv('../../data/customer_data.csv')
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(data[['fea_2']])
data[['fea_2']] = (imp.transform(data[['fea_2']]))

if __name__ == "__main__":

    qnt_experiments = 5
    neurons = 8
    val_size = 0.3
    losses_acc = []
    data_size = 0
    path = '../results/experiments - credit card 1.csv'

    for exp in range(1, qnt_experiments+1):

        tf.keras.backend.clear_session()
        model = Model(data, neurons)
        train_size = 0.5
        n_epochs = 50
        lr = 0.0001
        class_first = True

        experiment = Experiment(model,
                                n_epochs,
                                lr,
                                train_size,
                                val_size,
                                data,
                                class_first)

        experiment.fit()
        
        
        # err, acc = experiment.get_evaluate()
        
        # losses_acc.append([experiment.save_losses_val()[-1],
        #                 experiment.save_losses_train()[-1],
        #                 experiment.save_acuraccys_val()[-1],
        #                 experiment.save_acuraccys_train()[-1],
        #                 err,
        #                 acc])

        # data_size = experiment.get_train_size()

        # write_file(path, data_size, exp, neurons, model, losses_acc)
        experiment = None
        model = None
        print(f'{exp}/5')