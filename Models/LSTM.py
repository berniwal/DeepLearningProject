
import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

import CreateDataset

FILE_NAMES = ['algebra__linear_1d.txt']

BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 5000
LSTM_SIZE = 190

# Create Dataset of all FileNames with One Hot Character Encoding
# and Padding to LSTM_SIZE

all_labeled_data = CreateDataset.create_dataset(FILE_NAMES,LSTM_SIZE)

#Split in Train and Test Data (Remove .take(TAKE_SIZE) at train_data to train on all data)

train_data = all_labeled_data.skip(TAKE_SIZE).take(TAKE_SIZE)
train_data = train_data.batch(BATCH_SIZE)

test_data = all_labeled_data.take(TAKE_SIZE)
test_data = test_data.batch(BATCH_SIZE)

model = tf.keras.Sequential()
model.add(layers.Input(shape=(80, LSTM_SIZE)))
model.add(layers.LSTM(LSTM_SIZE, return_sequences=True, activation='sigmoid'))
print(model.summary())

optimizer = Adam(
    lr=6e-4,
    beta_1=0.9,
    beta_2=0.995,
    epsilon=1e-9,
    decay=0.0,
    amsgrad=False,
    clipnorm=0.1,
)

model.compile(optimizer=optimizer, loss="categorical_crossentropy")

model.fit(train_data,
          validation_data=test_data,
          epochs=3,
          verbose=1)

#Make sample prediction on one Test data Point

for x,y in train_data:
    print(x)git 
    y_pred = model.predict(x)
    print(y_pred)
    break

