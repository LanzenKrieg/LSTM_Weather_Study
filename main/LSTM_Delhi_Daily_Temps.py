import tensorflow as tf
import os
import pandas as pd
import numpy as np
from google.colab import drive
import matplotlib.pyplot as plt
import datetime
drive.mount('/content/drive')
path = "/content/drive/MyDrive"

file_path = os.path.join(path, 'DL_TEMP_DTS.csv')
df = pd.read_csv(file_path)



def dtm(df):
  Y = df['Year'].astype(int)
  M = df['Month'].astype(int)
  D = df['Day'].astype(int)
  return datetime.datetime(year = Y, month = M, day = D)

df['datetime']=df.apply(dtm, axis = 1)
df.pop('Year')
df.pop('Month')
df.pop('Day')
df.index = df.pop('datetime')
temp = df['Avg Daily Temp']
df

def df_To_X_Y(temp, window_size = 5):
  df_as_np = df.to_numpy()
  X = []
  Y = []
  for i in range(len(df_as_np) - window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size]
    Y.append(label)
  return np.array(X), np.array(Y)

WINDOW_SIZE = 5
X1, Y1 = df_To_X_Y(temp, WINDOW_SIZE)
X1.shape, Y1.shape

X_train, Y_train = X1[:4500], Y1[:4500]
X_val, Y_val = X1[4500:7500], Y1[4500:7500]
X_test, Y_test = X1[7500:], Y1[7500:]
X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

model1 = Sequential()
model1.add(InputLayer((WINDOW_SIZE, 1)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))

model1.summary()

cp1 = ModelCheckpoint('model1.keras', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

model1.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=15, callbacks=[cp1])

from tensorflow.keras.models import load_model
model1 = load_model('model1.keras')

train_predictions = model1.predict(X_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':Y_train.flatten()})
train_results

import matplotlib.pyplot as plt
plt.plot(train_results['Train Predictions'][250:500])
plt.plot(train_results['Actuals'][250:500])

val_predictions = model1.predict(X_val).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':Y_val.flatten()})
val_results

plt.plot(val_results['Val Predictions'][2500:])
plt.plot(val_results['Actuals'][2500:])

test_predictions = model1.predict(X_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':Y_test.flatten()})
test_results

plt.plot(test_results['Test Predictions'][1500:])
plt.plot(test_results['Actuals'][1500:])
