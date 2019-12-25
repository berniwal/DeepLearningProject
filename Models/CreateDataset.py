
import os

import tensorflow as tf
import tensorflow_datasets as tfds


#https://stackoverflow.com/questions/49370940/one-hot-encoding-characters
def convert_to_one_hot_tensor(data): # One Hot Encode the Characters

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

    return tf.transpose(tf.convert_to_tensor(one_hot))


def one_hot_encode_map(x):    # Map to Python Function from Tensorflow
    return (
        tf.py_function(convert_to_one_hot_tensor, inp=[x[0]], Tout=(tf.int32)),
        tf.py_function(convert_to_one_hot_tensor, inp=[x[1]], Tout=(tf.int32))
    )


def right_pad_tensor(tensor, desired_dimension):   # Right pad tensor with zeros to desired Dimension
    current_rows, current_cols = tf.shape(tensor)
    padding = tf.zeros([current_rows, desired_dimension - current_cols], dtype = tensor.dtype)
    tensor = tf.concat([tensor,padding], 1)
    return tensor


def left_pad_tensor(tensor,desired_dimension): # Left pad tensor with zeros to desired Dimension
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

    # Make two independent Tensors as Tuple (Not needed)

    # all_labeled_data = all_labeled_data.map(lambda x: (x[0], x[1]))

    # Shuffle the Data

    all_labeled_data = all_labeled_data.shuffle(
        BUFFER_SIZE,
        reshuffle_each_iteration=False
    )

    # Map all Datapoints to One Hot Labeled Datapoints Character Wise

    all_labeled_data = all_labeled_data.map(one_hot_encode_map)

    #Convert to good input dimension for LSTM

    all_labeled_data = all_labeled_data.map(
        lambda x, y: (
            tf.py_function(right_pad_tensor, inp=[x, LSTM_LENGTH], Tout=(tf.int32)),
            tf.py_function(left_pad_tensor, inp=[y, LSTM_LENGTH], Tout=(tf.int32))
        )
    )

    return all_labeled_data
