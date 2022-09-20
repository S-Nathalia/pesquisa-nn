from sklearn.model_selection import train_test_split
from functions import normalization
import tensorflow as tf
import numpy as np
import math

seed_val = 1
np.random.seed(seed_val)
tf.random.set_seed(seed_val)

class Experiment():

    def __init__(self, model, epochs, lr, train_size, val_size, data, class_first, qnt_data):
        self.model = model
        self.n_epochs = epochs
        self.lr = lr
        self.loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.metric_val = tf.keras.metrics.BinaryAccuracy()
        self.metric_train = tf.keras.metrics.BinaryAccuracy()
        self.metric_test = tf.keras.metrics.BinaryAccuracy()

        self.train_size = train_size
        self.val_size = val_size
        self.data = data
        self.qnt_data = qnt_data
        self.class_first = class_first
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = None, None, None, None, None, None
        self.load_dataset()
        self.batch_size = len(self.x_train)//100
        try:
            self.atualizations = math.ceil(len(self.y_train)/self.batch_size)
        except: 
            self.atualizations = 100

        self.losses_train = []
        self.acuraccy_train = []
        self.losses_val = []
        self.acuraccy_val = []

    def load_dataset(self):
        tam = self.val_size/(1-self.train_size)
        
        if self.class_first:
            x, y = self.data.iloc[:, 1:], self.data.iloc[:, :1]
        else:
            x, y = self.data.iloc[:, 1:], self.data.iloc[:, -1:]

        x_train, xtest, y_train, ytest = train_test_split(x, y, train_size=self.train_size, shuffle=True, stratify=y)
        x_train, xtest = normalization(x_train, xtest)
        x_val, x_test, y_val, y_test = train_test_split(xtest, ytest, train_size=tam, shuffle=True, stratify=ytest)
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
    
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            pred = self.model(x, training=True)
            err = self.loss(y, pred)

        grads = tape.gradient(err, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.metric_train.update_state(y, pred)
        return err
    
    @tf.function
    def val_step(self, x, y):
        pred = self.model(x, training=False)
        err = self.loss(y, pred)
        self.metric_val.update_state(y, pred)
        return err

    def fit(self):
        len_train = len(self.y_train)

        for epoch in range(self.n_epochs):
            epoch_train_loss = 0.0
            epoch_val_loss = 0.0

            for i in range(self.atualizations):
                x, y = [], []
                last_range = (i+1)*self.batch_size
                if (last_range >= len_train):
                    x = self.x_train[i*self.batch_size:]
                    y = self.y_train[i*self.batch_size:]
                else:
                    x = self.x_train[i*self.batch_size:last_range]
                    y = self.y_train[i*self.batch_size:last_range]

                epoch_train_loss += self.train_step(np.array(x), np.array(y)).numpy()

            epoch_val_loss = self.val_step(np.array(self.x_val), np.array(self.y_val)).numpy()
            
            epoch_train_loss /= self.atualizations


            if epoch%5 == 0:
                print(f'EPOCH {epoch} ' +
                    f'LOSS TRAIN: {epoch_train_loss:.4f} ACC TRAIN: {self.metric_train.result():.4f} ' +
                    f'LOSS VAL: {epoch_val_loss:.4f} ACC VAL: {self.metric_val.result():.4f}')

            self.losses_train.append(epoch_train_loss)
            self.losses_val.append(epoch_val_loss)
            self.acuraccy_train.append(self.metric_train.result().numpy())
            self.acuraccy_val.append(self.metric_val.result().numpy())

        self.metric_train.reset_states()
        self.metric_val.reset_states()


    def get_evaluate(self):
        pred = self.model(np.array(self.x_test), training=False)
        err = self.loss(np.array(self.y_test), pred).numpy()

        self.metric_test.update_state(np.array(self.y_test), pred)
        acc = self.metric_test.result().numpy()

        self.metric_test.reset_states()

        return err, acc

    def get_train_size(self):
        return len(self.x_train)

    def get_proportion(self):
        if self.class_first:
            y = self.data.iloc[:, :1]
        else:
            y = self.data.iloc[:, -1:]

        keys_qnts = np.unique(y, return_counts=True, axis=None)
        return keys_qnts

    def save_losses_val(self):
        return self.losses_val

    def save_losses_train(self):
        return self.losses_train

    def save_acuraccys_train(self):
        return self.acuraccy_train

    def save_acuraccys_val(self):
        return self.acuraccy_val