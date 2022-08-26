from functions import write_file
from Experiment import *
import tensorflow as tf
import pandas as pd
from Model import *
import numpy as np
import os

data = pd.read_csv('../../data/diabetes.csv')

if __name__ == "__main__":

    qnt_experiments = 6
    val_size = 0.3
    n_repeat = 5
    neurons = 8

    path = '../results/experiments - indian_diabets 1 '

    for exp in range(1, qnt_experiments+1):
        losses_val = []
        losses_train = []
        losses_test = []
        acc_val = []
        acc_train = []
        acc_test = []

        for n in range(n_repeat):
            model = None
            experiment = None
            print(f'{n}/{n_repeat}')
            print(losses_val)

            tf.keras.backend.clear_session()
            model = Model(data, neurons)
            train_size = exp/10
            n_epochs = 50
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
            
            err, acc = experiment.get_evaluate()
            
            losses_val.append(experiment.save_losses_val()[-1])
            losses_train.append(experiment.save_losses_train()[-1])
            losses_test.append(err)

        # path, experiment, array, string=''
        write_file(path, experiment, losses_val, string='loss_val')