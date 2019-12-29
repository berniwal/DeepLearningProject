
import os

import tensorflow as tf
#from tensorflow.keras import layers
from tensorflow.compat.v1.keras import layers
#from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.keras.optimizers import Adam

import CreateDataset

print(tf.__version__)

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
model.add(layers.LSTM(30, return_sequences=True))
model.add(layers.Dense(30))
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
    print(x[0])
    y_pred = model.predict(x)
    print(y_pred[0])

    alphabet = " abcdefghijklmnopqrstuvwxyz" \
               "ABCDEFGHIJKLMNOPQRSTUVWXYZ" \
               "0123456789*+-.=/()?,'>:<!{}"

    int_to_char = dict((i, c) for i, c in enumerate(alphabet))

    prediction = []
    for x in tf.argmax(y_pred[0], axis=1):
        prediction.append(int_to_char[x.numpy()])

    print(prediction)
    break

