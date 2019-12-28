
import os

import tensorflow as tf
import tensorflow_datasets as tfds


# https://stackoverflow.com/questions/49370940/one-hot-encoding-characters
def convert_to_one_hot_tensor(data): # One Hot Encode the Characters
    # Returns Tensor of Dimension (Alphabet_Length x String_Length)

    alphabet = " abcdefghijklmnopqrstuvwxyz" \
               "ABCDEFGHIJKLMNOPQRSTUVWXYZ" \
               "0123456789*+-.=/()?,'>:<!{}"

    data = data.numpy().decode("utf-8")

    # Creates a dict, that maps to every char of alphabet an unique int based on position
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    encoded_data = []

    # Replaces every char in data with the mapped int
    encoded_data.append([char_to_int[char] for char in data])

    encoded_data = encoded_data[0]

    # This part now replaces the int by an one-hot array with size alphabet
    one_hot = []

    for value in encoded_data:
        # At first, the whole array is initialized with 0
        letter = [0 for _ in range(len(alphabet))]
        # Only at the number of the int, 1 is written
        letter[value] = 1
        one_hot.append(letter)

    one_hot_tensor = tf.transpose(tf.convert_to_tensor(one_hot))

    return tf.dtypes.cast(one_hot_tensor, tf.float32)


def one_hot_encode_map(x, LSTM_LENGTH):    # Mapping Function can be applied on Dataset

    #first = convert_to_one_hot_tensor(x[0])
    first = tf.py_function(convert_to_one_hot_tensor, inp=[x[0]], Tout=(tf.float32))

    #second = convert_to_one_hot_tensor(x[1])
    second = tf.py_function(convert_to_one_hot_tensor,inp=[x[1]], Tout=(tf.float32))

    #Zero pad the Tensor to the LSTM Dimension
    first =  tf.py_function(right_pad_tensor, inp=[first,LSTM_LENGTH], Tout=(tf.float32))
    second = tf.py_function(left_pad_tensor, inp=[second,LSTM_LENGTH], Tout=(tf.float32))

    #Set the Shape as the Shape gets lost after py_function
    first.set_shape([80,190])
    second.set_shape([80,190])

    return first, second


def right_pad_tensor(tensor, desired_dimension):   # Right pad tensor with zeros to desired Dimension
    current_rows, current_cols = tf.shape(tensor)
    padding = tf.zeros([current_rows, desired_dimension - current_cols], dtype = tensor.dtype)
    tensor = tf.concat([tensor,padding], 1)
    return tensor


def left_pad_tensor(tensor,desired_dimension):    # Left pad tensor with zeros to desired Dimension
    current_rows, current_cols = tf.shape(tensor)
    padding = tf.zeros([current_rows, desired_dimension - current_cols], dtype=tensor.dtype)
    tensor = tf.concat([padding, tensor], 1)
    return tensor


def create_dataset(FILE_NAMES, LSTM_LENGTH):
    DIR_NAMES = ['train-easy/', 'train-medium/', 'train-hard']

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

    # Map all Datapoints to One Hot Labeled Datapoints Character Wise and pad to LSTM Length
    # Input gets left padded and Output gets Right Padded with Zeros
    all_labeled_data = all_labeled_data.map(lambda x: one_hot_encode_map(x,LSTM_LENGTH))

    return all_labeled_data
