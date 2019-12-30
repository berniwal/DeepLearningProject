
import os

import tensorflow as tf
#from tensorflow.keras import layers
from tensorflow.compat.v1.keras import layers
#from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.keras.optimizers import Adam

import CreateDataset

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

FILE_NAMES = ['algebra__linear_1d.txt']

BUFFER_SIZE = 50000
BATCH_SIZE = 256
TAKE_SIZE = 5000
LSTM_SIZE = 1024
TAKE_TRAIN_SIZE = 512000

total_matches = 0

@tf.function
def exact(true_label, predicted):
    pred_label = tf.argmax(tf.math.softmax(predicted, axis=2), axis=2)

    cost = 0
    pred_label = tf.dtypes.cast(pred_label, tf.float32)

    boolean_equal = tf.equal(true_label, pred_label)

    compute_any = tf.reduce_all(boolean_equal)

    if(compute_any):
        cost = 1

    return cost

# Create Dataset of all FileNames with One Hot Character Encoding
# and Padding to LSTM_SIZE

all_labeled_data = CreateDataset.create_dataset(FILE_NAMES,190)

#Split in Train and Test Data (Remove .take(TAKE_SIZE) at train_data to train on all data)

train_data = all_labeled_data.skip(TAKE_SIZE).take(TAKE_TRAIN_SIZE)
train_data = train_data.batch(BATCH_SIZE)

test_data = all_labeled_data.take(TAKE_SIZE)
test_data = test_data.batch(BATCH_SIZE)

print(train_data)

model = tf.keras.Sequential()
model.add(layers.Input(shape=(190,80)))
model.add(layers.LSTM(LSTM_SIZE, return_sequences=True))
model.add(layers.Dense(80))
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

#model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",from_logits=True, metrics=[exact])
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",from_logits=True)

checkpoint_path = "./checkpoints/cp-{epoch:04d}.ckpt"

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=1)

model.fit(train_data,
          validation_data=test_data,
          epochs=3,
          callbacks=[cp_callback],
          verbose=1)


