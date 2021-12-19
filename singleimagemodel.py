from logging import getLevelName
import numpy as np
import os
import tensorflow as tf
import pathlib
import pandas as pd
import re
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten , Conv1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D,MaxPooling1D
from tensorflow.keras.utils import plot_model

# Global config (TODO)
random_seed = 77
data_path = "./input/ocular-disease-recognition-odir5k/preprocessed_images/"
data_dir = pathlib.Path(data_path)
tf.config.run_functions_eagerly(True)



def load_sample_ids(df, val_size):
    sample_ids = df['ID'].to_list()
    dataset = tf.data.Dataset.from_tensor_slices(sample_ids)
    dataset = dataset.unique()

    dataset = dataset.shuffle(100)

    val_ds = dataset.take(val_size)
    test_ds = dataset.skip(val_size).take(val_size)
    train_ds = dataset.skip(2*val_size)

    return train_ds, val_ds, test_ds


def build_label_dictionary(df):
    dict = {}
    for index, row in df.iterrows():
        filename = row['filename']
        image_target = np.asarray(eval(row["target"]))
        dict[filename] = image_target
    return dict

def _extract_label(filename):
    return  label_dict[bytes.decode(filename.numpy())]


def get_label(filename):
    [label] = tf.py_function(_extract_label, [filename], [tf.int64])
    label.set_shape([8])
    return label

def _has_label(filename):
    return filename in label_dict

def has_label(filename):
    [label] = tf.py_function(_has_label, [filename], [tf.bool])
    label.set_shape([])
    return label


def _file_exists(file_path):
    return tf.io.gfile.exists(file_path.numpy())

def file_exists(file_path):
    [exists] = tf.py_function(_file_exists, [file_path], [tf.bool])
    exists.set_shape([])

    return exists

def filenames_from_id(id):
    right_path = tf.strings.as_string(id) + tf.constant("_right.jpg")
    left_path =  tf.strings.as_string(id) + tf.constant("_left.jpg")

    data = []
    for p in [left_path, right_path]:
        if file_exists(p):
            if has_label(p):
                img = p
                data.append(img)
    
    return tf.data.Dataset.from_tensor_slices(data)

def process_filename(filename):
    label = get_label(filename)
    img = filename
    return img, label

def print_dataset_stats(names, datasets):
    for name, dataset in zip(names, datasets):
        d = list(dataset.as_numpy_iterator())
        top5 = d[:5]
        print(f"{name} size: {len(d)} . First elements : {top5}")




df = pd.read_csv('./input/ocular-disease-recognition-odir5k/full_df.csv')
label_dict = build_label_dictionary(df)


train, val, test = load_sample_ids(df, 500)
loaded_train = train.flat_map(filenames_from_id).map(process_filename)



print_dataset_stats(["train", "val", "test", "loaded_train"],[train, val, test, loaded_train])

