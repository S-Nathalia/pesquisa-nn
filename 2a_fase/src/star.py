from functions import calculate_train_size as cal
from functions import write_file
from functions import load_dataset
from Experiment import *
import tensorflow as tf
import pandas as pd
from Model import *
# python3 heart.py & & python3 indiandiabets.py & & python3 neo.py python3 star.py & & python3 wine.py


data = pd.read_csv('../../data/star_att.csv')
path = '../results/experiments - star '

if __name__ == "__main__":

    val_size = 0.3
    n_repeat = 5
    qnt_data = 1500
    x_train, y_train, x_val, y_val, x_test, y_test = None, None, None, None, None, None

    while (qnt_data < 1550):
        qnt_data += 50
        class_first = False
        train_size = cal(data, qnt_data)
        change_samples = False
        x_train, y_train, x_val, y_val, x_test, y_test = load_dataset(
            class_first, change_samples, data, val_size, train_size, qnt_data)

        for neurons in [500, 550, 600]:
            losses_val = []
            losses_train = []
            losses_test = []
            acc_val = []
            acc_train = []
            acc_test = []

            for n in range(n_repeat):
                tf.keras.backend.clear_session()
                model = None
                experiment = None
                model = Model(data, neurons)
                n_epochs = 50
                lr = 0.0001
                change_samples = True
                x_train, y_train, x_val, y_val, x_test, y_test = load_dataset(
                    class_first, change_samples, data, val_size, train_size, qnt_data,
                    x_train, y_train, x_val, y_val, x_test, y_test)

                experiment = Experiment(model,
                                        n_epochs,
                                        lr,
                                        train_size,
                                        val_size,
                                        data,
                                        class_first,
                                        x_train, y_train, x_val, y_val, x_test, y_test)

                experiment.fit()

                err, acc = experiment.get_evaluate()

                losses_val.append(experiment.save_losses_val()[-1])
                losses_train.append(experiment.save_losses_train()[-1])
                losses_test.append(err)
                acc_val.append(experiment.save_acuraccys_val()[-1])
                acc_train.append(experiment.save_acuraccys_val()[-1])
                acc_test.append(acc)

            # path, experiment, array, string=''
            write_file(path, experiment, losses_val, string='loss_val')
            write_file(path, experiment, losses_train, string='loss_train')
            write_file(path, experiment, losses_test, string='loss_test')

            write_file(path, experiment, acc_val, string='acc_val')
            write_file(path, experiment, acc_train, string='acc_train')
            write_file(path, experiment, acc_test, string='acc_test')
