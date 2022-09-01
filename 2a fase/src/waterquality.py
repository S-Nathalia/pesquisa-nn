from functions import calculate_train_size as cal
from functions import write_file
from Experiment import *
import tensorflow as tf
import pandas as pd
from Model import *
import numpy as np

data = pd.read_csv('../../data/water_potability.csv')
path = '../results/experiments - water quality 1 '
data.dropna(axis=0, inplace=True)


if __name__ == "__main__":

    val_size = 0.3
    n_repeat = 5
    qnt_data = 0


    for neurons in [256]:

        while(qnt_data < 12000):
            qnt_data += 300
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
                train_size = cal(data, qnt_data)
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
            write_file(path, experiment, losses_val, string='loss_train')
            write_file(path, experiment, losses_val, string='loss_test')
            write_file(path, experiment, losses_val, string='acc_val')
            write_file(path, experiment, losses_val, string='acc_train')
            write_file(path, experiment, losses_val, string='acc_test')