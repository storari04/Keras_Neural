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
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

fold_number = 5
trials = 40
fraction_of_validation_samples = 0.2
number_of_sub_models = 500

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

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])
model.summary()

early_stop = keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)

history = model.fit(autocalculated_train_x_np, train_y,
                    batch_size=128,
                    epochs=30,
                    verbose=0,
                    validation_split = 0.2,
                    callbacks=[early_stop])

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
