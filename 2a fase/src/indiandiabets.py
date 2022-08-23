from Experiment import *
import tensorflow as tf
import pandas as pd
import numpy as np
from Model import *
import os



def write_file(path, data_size, exp, neurons, model, array):
    array = np.array(array)

    with open(path+'values', 'a') as file:
        file.write(f'{exp}\n')
        file.write(f'{data_size}\n')
        file.write(f'{model.neurons}\n')
        file.write(f'{count_weights(model)}\n')
        file.write(f'{array} \n')

    with open(path, 'a') as file:

        file.write(f'{exp},')
        file.write(f'{data_size},')
        file.write(f'{model.neurons},')
        file.write(f'{count_weights(model)},')
        for i in range(6):
            file.write(f'{array[:, i].sum()/len(array)},')
        file.write('\n')
        file.close()

data = pd.read_csv('../../data/diabetes.csv')

if __name__ == "__main__":

    qnt_experiments = 6
    val_size = 0.3
    n_repeat = 2
    data_size = 0
    neurons = 10
    path = '../results/experiments - indian diabets 1.csv'

    # for exp in range(1, qnt_experiments+1):
    #     if exp%5 == 0 or exp == 1:
    #         neurons = 1

    #         for k in range(1):
    #             neurons *= 2
    #             mean_data = []

    for rpt in range(n_repeat):
        print(f'\n\n{rpt+1}/{n_repeat}')


        model = Model(data, neurons)
        # train_size = exp/10
        train_size = 0.5
        n_epochs = 20
        lr = 0.0001
        class_first = False

        experiment = Experiment(model,
                                n_epochs,
                                lr,
                                train_size,
                                val_size,
                                data,
                                class_first)

        experiment.fit()
        model = 'a'
        experiment = 'b'
        tf.keras.backend.clear_session()
                    # err, acc = experiment.get_evaluate()

                    # mean_data.append([experiment.save_losses_val()[-1],
                    #                 experiment.save_losses_train()[-1],
                    #                 err,
                    #                 experiment.save_acuraccys_val()[-1],
                    #                 experiment.save_acuraccys_train()[-1],                        
                    #                 acc])
                    # data_size = experiment.get_train_size()
                
                # write_file(path, data_size, exp, neurons, model, mean_data)
