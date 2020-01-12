#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import string

import tensorflow as tf
import tensorflow.keras as keras
#from tensorflow.keras import layers
from tensorflow.compat.v1.keras import layers
#from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.keras.optimizers import Adam


# In[2]:


DIR_NAMES = ['train-easy/']
FILE_NAMES = ['algebra__linear_1d.txt']

BUFFER_SIZE = 50000

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
dataset_dir = parent_dir + '/Dataset'


# In[3]:


# Based on https://www.tensorflow.org/tutorials/load_data/text
# Read in all files which are in FILE_NAMES
labeled_data_sets = []

for file_name in FILE_NAMES:
    for dir_name in DIR_NAMES:
        concat_dir = os.path.join(dir_name, file_name)
        lines_dataset = tf.data.TextLineDataset(
            os.path.join(dataset_dir, concat_dir)
        )
        labeled_data_sets.append(lines_dataset)


# In[4]:


# Concatenate all File Data to one Big File Data
all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)


# In[5]:


# Group Data as batches of two (input_sentence, answer)
all_labeled_data = all_labeled_data.batch(2)


# In[6]:


# Shuffle the Data
all_labeled_data = all_labeled_data.shuffle(
        BUFFER_SIZE,
        reshuffle_each_iteration=False
)


# In[7]:


x, y = next(iter(all_labeled_data))
print(next(iter(all_labeled_data)))


# In[8]:


print(x)
print(y)


# In[9]:


#Same Code as in Data Generation
MAX_QUESTION_LENGTH = 160
MAX_ANSWER_LENGTH = 30
LSTM_LENGTH = 250
QUESTION_CHARS = ( ['', ' '] + list(string.ascii_letters + string.digits + string.punctuation))
CHAR_TO_INDEX = {char: index for index, char in enumerate(QUESTION_CHARS)}
INDEX_TO_CHAR = {index: char for index, char in enumerate(QUESTION_CHARS)}

NUM_INDICES = len(QUESTION_CHARS)

print(CHAR_TO_INDEX)

keras.utils.to_categorical(0, num_classes=NUM_INDICES, dtype='float32')


# In[10]:


def map_encode_data(x):
    x_one_hot = tf.py_function(encode_data, inp=[x[0]], Tout=(tf.float32))
    y_one_hot = tf.py_function(encode_data, inp=[x[1]], Tout=(tf.float32))
    y_number = tf.py_function(num_labels, inp=[x[1]], Tout=(tf.float32))

    x_one_hot = tf.py_function(input_padding, inp=[x_one_hot, y_one_hot, LSTM_LENGTH], Tout=(tf.float32))
    y_one_hot = tf.py_function(output_padding, inp=[y_one_hot, MAX_ANSWER_LENGTH, LSTM_LENGTH], Tout=(tf.float32))
    y_number = tf.py_function(output_padding_number, inp=[y_number, MAX_ANSWER_LENGTH, LSTM_LENGTH], Tout=(tf.float32))
    
    x_one_hot.set_shape([LSTM_LENGTH,NUM_INDICES])
    y_one_hot.set_shape([LSTM_LENGTH,NUM_INDICES])
    y_number.set_shape([LSTM_LENGTH])
    
    return x_one_hot, y_number
    

def encode_data(x):
    x_encoded = [CHAR_TO_INDEX[z] for z in x.numpy().decode('utf-8')]
    x_one_hot = keras.utils.to_categorical(x_encoded, num_classes=NUM_INDICES, dtype='float32')
    
    return x_one_hot

def input_padding(tensor, out_tensor, desired_dimension):   
    # Right pad tensor with zeros and out_tensor shifted by right to desired Dimension
    
    current_rows, current_cols = tf.shape(tensor)
    current_rows_out, current_cols_out = tf.shape(out_tensor)
    padding = tf.zeros([desired_dimension - current_rows -current_rows_out + 1, current_cols], dtype = tensor.dtype)
    tensor = tf.concat([tensor,padding,out_tensor[:-1]], 0)
    return tensor

def output_padding(tensor, output_dimension, desired_dimension):   # Right pad tensor with zeros to desired Dimension
    current_rows, current_cols = tf.shape(tensor)
    if(current_rows - output_dimension != 0):
        padding_right = tf.zeros([output_dimension - current_rows, NUM_INDICES-1], dtype = tensor.dtype)
        padding = tf.zeros([desired_dimension - output_dimension, NUM_INDICES-1], dtype = tensor.dtype)
        padding_right_ones = tf.ones([output_dimension - current_rows, 1], dtype = tensor.dtype)
        padding_ones = tf.ones([desired_dimension - output_dimension, 1], dtype = tensor.dtype)
        
        padding = tf.concat([padding_ones, padding], 1)
        padding_right = tf.concat([padding_right_ones, padding_right], 1)
        
        tensor = tf.concat([padding,tensor,padding_right], 0)
    else:
        padding = tf.zeros([desired_dimension - output_dimension, NUM_INDICES-1], dtype=tensor.dtype)
        padding_ones = tf.ones([desired_dimension - output_dimension, 1], dtype = tensor.dtype)
        padding = tf.concat([padding_ones, padding], 1)
        tensor = tf.concat([padding,tensor], 0)
    return tensor


def output_padding_number(tensor, output_dimension, desired_dimension):   # Right pad tensor with zeros to desired Dimension
    current_rows = tf.shape(tensor)
    if(current_rows - output_dimension != 0):
        padding_right = tf.zeros([output_dimension - current_rows], dtype = tensor.dtype)
        padding = tf.zeros([desired_dimension - output_dimension], dtype = tensor.dtype)
        tensor = tf.concat([padding,tensor,padding_right], 0)
    else:
        padding = tf.zeros([desired_dimension - output_dimension], dtype=tensor.dtype)
        tensor = tf.concat([padding,tensor], 0)
    return tensor


def num_labels(data):

    data = data.numpy().decode("utf-8")

    encoded_data = []

    # Replaces every char in data with the mapped int
    encoded_data.append([CHAR_TO_INDEX[z] for z in data])

    encoded_data = tf.convert_to_tensor(encoded_data[0])
    return tf.dtypes.cast(encoded_data, tf.float32)

# In[11]:


all_labeled_data = all_labeled_data.map(lambda x: map_encode_data(x))


# In[12]:


BATCH_SIZE = 256
TAKE_SIZE = 5000
TAKE_TRAIN_SIZE = 5000


# In[ ]:


#Split in Train and Test Data (Remove .take(TAKE_SIZE) at train_data to train on all data)

train_data = all_labeled_data.skip(TAKE_SIZE).take(TAKE_TRAIN_SIZE)
train_data = train_data.batch(BATCH_SIZE)

test_data = all_labeled_data.take(TAKE_SIZE)
test_data = test_data.batch(BATCH_SIZE)

print(train_data)

model = tf.keras.Sequential()
model.add(layers.Input(shape=(LSTM_LENGTH,NUM_INDICES)))
model.add(layers.LSTM(LSTM_LENGTH, return_sequences=True))
model.add(layers.Dense(NUM_INDICES))
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

checkpoint_dir = "./checkpoints/"
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)
model.load_weights(latest)

# Make sample prediction on one Test data Point

for x, y in test_data:
    y_pred = model.predict(x)

    for i in range(5):
        prediction = []
        for x in tf.argmax(tf.math.softmax(y_pred[0], axis=1), axis=1):
            prediction.append(INDEX_TO_CHAR[x.numpy()])

        y_true = []
        for x in y[0]:
            y_true.append(INDEX_TO_CHAR[x.numpy()])

        print(y_true)
        print(prediction)

    break