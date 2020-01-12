#!/usr/bin/env python
# coding: utf-8

# In[93]:


#https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np

import os
import string


# In[80]:


DIR_NAMES = ['train-easy/']
FILE_NAMES = ['algebra__linear_1d.txt']

BUFFER_SIZE = 50000

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
dataset_dir = parent_dir + '/Dataset'

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
        
# Concatenate all File Data to one Big File Data
all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
    
# Group Data as batches of two (input_sentence, answer)
all_labeled_data = all_labeled_data.batch(2)

# Shuffle the Data
all_labeled_data = all_labeled_data.shuffle(
        BUFFER_SIZE,
        reshuffle_each_iteration=False
)

#Same Code as in Data Generation
MAX_QUESTION_LENGTH = 160
MAX_ANSWER_LENGTH = 30
ENCODING_LSTM_LENGTH = 512
DECODING_LSTM_LENGTH = 2048
QUESTION_CHARS = ( ['', ' '] + list(string.ascii_letters + string.digits + string.punctuation))
CHAR_TO_INDEX = {char: index for index, char in enumerate(QUESTION_CHARS)}
INDEX_TO_CHAR = {index: char for index, char in enumerate(QUESTION_CHARS)}

NUM_INDICES = len(QUESTION_CHARS)

def encode_data(x):
    x_encoded = [CHAR_TO_INDEX[z] for z in x.numpy().decode('utf-8')]
    x_one_hot = keras.utils.to_categorical(x_encoded, num_classes=NUM_INDICES, dtype='float32')
    
    return x_one_hot

def input_padding(tensor, desired_dimension, shift):   

    current_rows, current_cols = tf.shape(tensor)
    
    if(desired_dimension - current_rows > 0):
        padding = tf.zeros([desired_dimension - current_rows,NUM_INDICES - 1], dtype = tensor.dtype)
        padding_ones = tf.ones([desired_dimension - current_rows, 1], dtype =tensor.dtype)
        complete_padding = tf.concat([padding_ones, padding], 1)
        tensor = tf.concat([tensor, complete_padding], 0)
        
    if(shift):
        shift_tensor = tf.zeros([shift, NUM_INDICES -1], dtype = tensor.dtype)
        shift_tensor_ones = tf.ones([shift, 1], dtype = tensor.dtype)
        shift_tensor = tf.concat([shift_tensor_ones, shift_tensor], 1)
        tensor = tf.concat([shift_tensor, tensor[:-shift]], 0)
        
    return tensor

def map_encode_data(x, reqlength, shift, index):
    x_one_hot = tf.py_function(encode_data, inp=[x[index]], Tout=(tf.float32))
    
    x_one_hot = tf.py_function(input_padding, inp=[x_one_hot, reqlength, shift], Tout=(tf.float32))
    
    x_one_hot.set_shape([reqlength,NUM_INDICES])
    
    return x_one_hot

encoder_input_data = all_labeled_data.map(lambda x: map_encode_data(x, MAX_QUESTION_LENGTH, 0, 0))

decoder_input_data = all_labeled_data.map(lambda x: map_encode_data(x, MAX_ANSWER_LENGTH, 1, 1))

decoder_target_data = all_labeled_data.map(lambda x: map_encode_data(x, MAX_ANSWER_LENGTH, 0, 1))

dataset_input = tf.data.Dataset.zip((encoder_input_data, decoder_input_data))

dataset_to_take = tf.data.Dataset.zip((dataset_input, decoder_target_data))

dataset = dataset_to_take.skip(5000).take(100000).batch(1024)

dataset_validation = dataset_to_take.take(5000).batch(1024)

print(dataset)
print(dataset_validation)


# In[81]:


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, NUM_INDICES))
encoder = LSTM(ENCODING_LSTM_LENGTH, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, NUM_INDICES))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the 
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(ENCODING_LSTM_LENGTH, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(NUM_INDICES, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

checkpoint_dir = "./checkpoints_attentional/"
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)
model.load_weights(latest)

# In[105]:


encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(ENCODING_LSTM_LENGTH,))
decoder_state_input_c = Input(shape=(ENCODING_LSTM_LENGTH,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


# In[113]:


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, NUM_INDICES))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, 0] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = INDEX_TO_CHAR[sampled_token_index]
        decoded_sentence += sampled_char
        
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '' or
           len(decoded_sentence) > MAX_ANSWER_LENGTH):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, NUM_INDICES))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


# In[117]:


x = encoder_input_data.take(1).batch(1)
print(next(iter(all_labeled_data)))
print(decode_sequence(x))





