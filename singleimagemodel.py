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
data_path_tensor = tf.constant(data_path)
data_dir = pathlib.Path(data_path)
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 16
img_height = 224
img_width = 224
class_count = 8
image_channels = 3

tf.config.run_functions_eagerly(True)



def load_sample_ids(df, val_size):
    sample_ids = df['ID'].to_list()
    dataset = tf.data.Dataset.from_tensor_slices(sample_ids)
    dataset = dataset.unique()

    dataset = dataset.shuffle(len(sample_ids))

    val_ds = dataset.take(val_size)
    test_ds = dataset.skip(val_size).take(val_size)
    train_ds = dataset.skip(2*val_size)

    return train_ds, val_ds, test_ds


def build_label_dictionary(df):
    keys = []
    values = []
    for index, row in df.iterrows():
        filename = row['filename']
        target = eval(row["target"])
        image_target = next(i for i,v in enumerate(target) if v==1)
        keys.append(filename)
        values.append(image_target)

    keys_tensor = tf.constant(keys)
    vals_tensor = tf.constant(values)
    table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), default_value=-1)

    return table

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
    return tf.io.gfile.exists(data_path + bytes.decode(file_path.numpy()))

def file_exists(file_path):
    [exists] = tf.py_function(_file_exists, [file_path], [tf.bool])
    exists.set_shape([])

    return exists

def filenames_from_id(id):
    right_path = tf.strings.as_string(id) + tf.constant("_right.jpg")
    left_path =  tf.strings.as_string(id) + tf.constant("_left.jpg")
    return tf.data.Dataset.from_tensor_slices([left_path, right_path])

def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=image_channels)
    return tf.image.resize(img, [img_height, img_width])

def process_filename(filename):
    img = tf.io.read_file(tf.strings.join([data_path_tensor, filename], ''))
    img = decode_img(img)
    return img, label_dict.lookup(filename)

def print_dataset_stats(names, datasets, n=-1):
    for name, dataset in zip(names, datasets):
        if n>0:
            dataset = dataset.take(n)

        d = list(dataset.as_numpy_iterator())
        

        top5 = d[:5]
        print(f"{name} size: {len(d)} . First elements : {top5}")

def label_not_missing(data, label):
    return tf.math.not_equal(label,-1)

def prepare_data(ds):
    filenames = ds.flat_map(filenames_from_id)
    existing_files = filenames.filter(file_exists)
    existing_files_and_labels = existing_files.map(process_filename)
    existing_files_and_existing_labels = existing_files_and_labels.filter(label_not_missing)
    data_and_labels = existing_files_and_existing_labels.map(lambda x,y : (x, tf.one_hot(y,class_count)))
    return data_and_labels

def configure_for_performance(ds):
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds


def create_model():
    inp1 = Input(shape=(img_height,img_width,image_channels), name="img")
    new_input = Input(shape=(img_height,img_width, image_channels), name="New Input")

    conv1 = Conv2D(3, kernel_size=3, padding ='same', activation='relu', name="conleft1")(inp1)
    i1 = tf.keras.applications.ResNet50(include_top=False,weights="imagenet",input_tensor=new_input,input_shape=None, pooling='avg')(conv1)

    class1 = Dense(1024, activation='relu')(i1)
    class1 = Dense(256, activation='relu')(class1)
    class1 = Dense(64, activation='relu')(class1)
    output = Dense(class_count, activation='sigmoid')(class1)
    model = Model(inputs=[inp1], outputs=output)
    return model

def train_model(model, training_data, validation_data):
    model = create_model()
    METRICS = [
        'accuracy',
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
    ]  
    model.compile(
        optimizer='Adam',
        loss='binary_crossentropy',
        metrics=METRICS
    )

    tf.keras.utils.plot_model(
        model,
        to_file="model.png",
        show_shapes=True,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=300,
        layer_range=None,
    )


    model.fit(
    training_data,
    validation_data=validation_data,
    epochs=50
    )


df = pd.read_csv('./input/ocular-disease-recognition-odir5k/full_df.csv')
label_dict = build_label_dictionary(df)


train, val, test = load_sample_ids(df, 500)

training_data = configure_for_performance(prepare_data(train))
validation_data = configure_for_performance(prepare_data(val))
test_data = configure_for_performance(prepare_data(test))
print(training_data.element_spec)

train_model(create_model(), training_data, validation_data)


# print_dataset_stats(["training_data"],[training_data],5)

