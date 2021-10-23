import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras


def make_dataset(text):
    '''
    Make a tensorflow dataset using text

    Parameters
    ----------
    text : str
        The data on which the model is trained

    Returns
    -------
    dataset : tensorflow.data.Dataset
        A dataset generated from the text input
    
    batch_size : int
        The input batch_size

    word_count : int
        The count of unique words in the dataset
        
    '''

    tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts([text])
    [encoded] = np.array(tokenizer.texts_to_sequences([text])) - 1
    print("Shape of encoded characters : ", encoded.shape)
    document_count = tokenizer.document_count
    print("Document size : ", document_count)
    word_count = tokenizer.word_counts
    print("Unique characters : ", len(word_count))
    '''
    print("Token count : ", word_count, "\n")
    word_index = tokenizer.word_index
    print("Index of tokens", word_index, "\n")
    '''

    @tf.autograph.experimental.do_not_convert
    def mapper(windows):
        return windows[:, :-1], windows[:, 1:]

    def encoder(X_batch, Y_batch):
        return tf.one_hot(X_batch, depth=len(word_count)), Y_batch

    n_steps = 100
    window_length = n_steps+1
    batch_size = 128

    dataset = tf.data.Dataset.from_tensor_slices(encoded)
    dataset = dataset.window(window_length, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_length))
    dataset = dataset.shuffle(10000).batch(batch_size)
    dataset = dataset.map(mapper)
    dataset = dataset.map(encoder)
    dataset = dataset.prefetch(1)

    return dataset, batch_size, word_count


def find_base(filename):
    checkpoints_filepath = os.path.join(
        os.curdir, f"checkpoints_{filename}"
    )
    directory = os.listdir(checkpoints_filepath)
    print("Directory listing : ", directory)
    base_model = directory[int(input('Enter position of model : '))]
    return base_model


def loading_path(filename, base_model):
    '''
    Generates the filepath from where the model would be loaded

    Parameters
    ----------
    filename : str
        The name of the file with which the model would be trained

    base_model : str
        The basic model which is to be trained or retrained

    Returns
    -------
    load_filepath : str
        The filepath for loading model
    
    '''

    root_filepath = os.path.join(
        os.curdir, f"checkpoints_{filename}\\{base_model}"
    )
    file_name = os.listdir(root_filepath)[-1]
    load_filepath = f"{root_filepath}\\{file_name}"
    print("Loading path : ", load_filepath)
    return load_filepath


def logging_path(filename, base_model):
    '''
    Generates the filepath where the model training data should be logged

    Parameters
    ----------
    filename : str
        The name of the file with which the model would be trained

    base_model : str
        The basic model which is to be trained or retrained

    Returns
    -------
    log_dir : str
        The filepath for logging training data
    
    '''

    run_id = time.strftime("date_%Y_%m_%d-time_%H_%M_%S")
    log_dir = os.path.join(
        os.curdir, f"logs_{filename}\\{base_model}\\{run_id}"
    )
    print("Logging directory : ", log_dir)
    return log_dir


def saving_path(filename, mode, base_model, model_name):
    '''
    Generates the filepath where the model should be saved

    Parameters
    ----------
    filename : str
        The name of the file with which the model would be trained

    mode : str ['train', 'retrain']
        The mode on which the program is run

    base_model : str
        The basic model which is to be trained or retrained

    model_name : str
        The actual model name with which the model is saved
    
    Returns
    -------
    save_filepath : str
        The filepath for saving trained model
    
    '''

    root_filepath = os.path.join(os.curdir, f"checkpoints_{filename}")
    if mode == "train":
        os.mkdir(f"{root_filepath}\\{base_model}")
    save_filepath = os.path.join(
        os.curdir, f"checkpoints_{filename}\\{base_model}\\{model_name}.h5"
    )
    print("Saving directory : ", save_filepath)
    return save_filepath
