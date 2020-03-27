import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
from sklearn import model_selection, metrics
from sklearn.model_selection import GridSearchCV, train_test_split
import math
import optuna
from sklearn.preprocessing import LabelEncoder
import re
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import RMSprop

fold_number = 5
trials = 40
fraction_of_validation_samples = 0.2
number_of_sub_models = 500

#autoscaling for normal
#MNIST
(train_x, train_y), (test_x, test_y) = mnist.load_data()

#autoscaling for normal
#autocalculated_train_x = (train_x - train_x.mean(axis = 0))/ train_x.std(ddof = 1, axis = 0)
#autocalculated_test_x = (test_x - train_x.mean(axis = 0))/ train_x.std(ddof = 1, axis = 0)

#autocalculated_train_x_np = autocalculated_train_x.values
#autocalculated_test_x_np = autocalculated_test_x.values

autocalculated_train_x_np = train_x.reshape(60000, 784)
autocalculated_test_x_np = test_x.reshape(10000, 784)
autocalculated_train_x_np =autocalculated_train_x_np.astype('float32')
autocalculated_test_x_np = autocalculated_test_x_np.astype('float32')
autocalculated_train_x_np /= 255
autocalculated_test_x_np /= 255
train_y = keras.utils.to_categorical(train_y, 10)
test_y = keras.utils.to_categorical(test_y, 10)


# convert class vectors to binary class matrices
import tensorflow as tf
# autocalculated_train_y_np = keras.utils.np_utils.to_categorical(autocalculated_train_y_np,2)

def create_model(num_layers, activation, mid_units, learning_rate, optimizer):
    model = Sequential()
    model.add(Dense(1024, activation = activation, input_shape = (784,)))

    for i in range(num_layers):
        model.add(Dense(128, activation = activation))
        #model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())

    model.add(Dense(10, activation='softmax'))

    return model

def objective(trial):
    optimizer = trial.suggest_categorical('optimizer', ['sgd', 'Adam', 'RMSprop'])
    activation = trial.suggest_categorical('activation', ['relu', 'sigmoid'])
    mid_units = int(trial.suggest_discrete_uniform('mid_units', 100, 500, 100))
    num_layers = trial.suggest_int('num_layers', 2, 4)
    #dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 1.0)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    model = create_model(num_layers, activation, mid_units, learning_rate, optimizer)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    early_stop = keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)
    history = model.fit(autocalculated_train_x_np, train_y, verbose = 0, epochs = 10, batch_size = 128, validation_split = 0.2, callbacks = [early_stop])
    return 1 - history.history['val_acc'][-1]

study = optuna.create_study()
study.optimize(objective, n_trials = trials)

print(study.best_params)
print(study.best_value)
best_optimizer = study.best_params['optimizer']
print(best_optimizer)

early_stop = keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)
best_model = create_model(**study.best_params)
best_model.compile(optimizer = best_optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
best_history = best_model.fit(autocalculated_train_x_np, train_y,
                    batch_size=128,
                    epochs=10,
                    verbose=0,
                    validation_split = 0.2,
                    callbacks=[early_stop])

train_acc = best_history.history['loss']
test_acc = best_history.history['val_loss']
x = np.arange(len(train_acc))
plt.plot(x, train_acc, label = 'train mse')
plt.plot(x, test_acc, label = 'test mse')
plt.legend() #グラフの線の説明を表示
plt.show()
