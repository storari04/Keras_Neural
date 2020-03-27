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
from sklearn.preprocessing import LabelEncoder
import optuna
import re
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.datasets import load_diabetes

fold_number = 5
trials = 5
fraction_of_validation_samples = 0.2
number_of_sub_models = 500

# use diabetes sample data from sklearn
diabetes = load_diabetes()

# load them to X and Y
X = diabetes.data
Y = diabetes.target

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1, random_state=0)

# convert class vectors to binary class matrices
import tensorflow as tf
# autocalculated_train_y_np = keras.utils.np_utils.to_categorical(autocalculated_train_y_np,2)

def create_model(num_layers, activation, mid_units, learning_rate, optimizer):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim = 10))

    for i in range(num_layers):
        model.add(Dense(16, activation = activation))
        #model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())

    model.add(Dense(1, activation='softmax'))

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
    estimator = KerasRegressor(build_fn = model, epochs = 5, batch_size = 16, verbose = 0)
    estimated_y_in_cv = model_selection.cross_val_predict(estimator, train_x, train_y, cv=fold_number)
    r2 = metrics.r2_score(train_y, estimated_y_in_cv)
    return 1.0 - r2

    #estimator.fit(train_x, train_y)
    #return 1 - estimator.history['val_acc'][-1]
    #history = model.fit(train_x, train_y, verbose = 0, epochs = 10, batch_size = 128, validation_split = 0.2, callbacks = [early_stop])
    #return 1 - history.history['val_acc'][-1]

study = optuna.create_study()
study.optimize(objective, n_trials = trials)

print(study.best_params)
print(study.best_value)
best_optimizer = study.best_params['optimizer']
print(best_optimizer)

early_stop = keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)
estimator = KerasRegressor(build_fn = create_model(**study.best_params), epochs = 5, batch_size = 16, verbose = 0)
estimator_history = estimator.fit(train_x, train_y)

results = model_selection.cross_val_score(estimator, train_x, train_y, cv= 5)
print(results)


train_acc = estimator_history.history['loss']
test_acc = estimator_history.history['val_loss']
x = np.arange(len(train_acc))
plt.plot(x, train_acc, label = 'train mse')
plt.plot(x, test_acc, label = 'test mse')
plt.legend() #グラフの線の説明を表示
plt.show()



"""
train_acc = history.history['loss']
test_acc = history.history['val_loss']
x = np.arange(len(train_acc))
plt.plot(x, train_acc, label = 'train accuracy')
plt.plot(x, test_acc, label = 'test accuracy')
plt.legend() #グラフの線の説明を表示
plt.show()

#calculated, estimated
calculated_ytrain = np.ndarray.flatten(model.predict(autocalculated_train_x))
calculated_ytrain = calculated_ytrain * train_y.std(ddof = 1, axis = 0) + train_y.mean(axis = 0)

# r2, RMSE, MAE
print('r2: {0}'.format(float(1 - sum((train_y - calculated_ytrain) ** 2) / sum((train_y - train_y.mean()) ** 2))))
print('RMSE: {0}'.format(float((sum((train_y - calculated_ytrain) ** 2) / len(train_y)) ** 0.5)))
print('MAE: {0}'.format(float(sum(abs(train_y - calculated_ytrain)) / len(train_y))))

##prediction
predicted_ytest = np.ndarray.flatten(model.predict(autocalculated_test_x))
predicted_ytest = predicted_ytest * train_y.std(ddof = 1, axis = 0) + train_y.mean(axis = 0)

#submit file
#submit_file_estimated = pd.DataFrame({'id' : train_id , 'kane' : estimated_ytrain})
#submit_file_estimated.to_csv('TakedaTargetKeras_estimated_1.00.tsv', index = False, header = False, sep = '\t')

#submit file
#submit_file = pd.DataFrame({'id' : test_id , 'kane' : predicted_ytest})
#submit_file.to_csv('TakedaTargetKeras_1.00.tsv', index = False, header = False, sep = '\t')
"""
