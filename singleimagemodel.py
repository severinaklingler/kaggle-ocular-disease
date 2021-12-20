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
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import datetime

import argparse

# Global config (TODO)
random_seed = 77
data_path = "./input/ocular-disease-recognition-odir5k/preprocessed_images/"
data_path_tensor = tf.constant(data_path)
data_dir = pathlib.Path(data_path)
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
img_height = 224
img_width = 224
class_count = 8
image_channels = 3
num_threads = 4
label_dict = {}

# tf.config.run_functions_eagerly(True)



def load_sample_ids(df, val_size):
    sample_ids = df['ID'].to_list()
    dataset = tf.data.Dataset.from_tensor_slices(sample_ids)
    dataset = dataset.unique()

    dataset = dataset.shuffle(len(sample_ids))

    val_ds = dataset.take(val_size)
    test_ds = dataset.skip(val_size).take(val_size)
    train_ds = dataset.skip(2*val_size)

    return train_ds, val_ds, test_ds

def decode_one_hot(x):
    return next(i for i,v in enumerate(x) if v==1)

def build_label_dictionary(df):
    keys = []
    values = []
    for index, row in df.iterrows():
        filename = row['filename']
        target = eval(row["target"])
        image_target = decode_one_hot(target)
        keys.append(filename)
        values.append(image_target)

    keys_tensor = tf.constant(keys)
    vals_tensor = tf.constant(values)
    table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), default_value=-1)

    return table

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
    existing_files_and_labels = existing_files.map(process_filename, num_parallel_calls=num_threads)
    existing_files_and_existing_labels = existing_files_and_labels.filter(label_not_missing)
    data_and_labels = existing_files_and_existing_labels.map(lambda x,y : (x, tf.one_hot(y,class_count)), num_parallel_calls=num_threads)
    return data_and_labels

def configure_for_performance(ds):
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=1)
  return ds

def show_batch(ds):
    images_batch, label_batch = next(iter(ds))

    plt.figure(figsize=(10, 10))
    for i in range(8):
        ax = plt.subplot(2, 4, i + 1)
        label = label_batch[i]
        print("Image shape: ", images_batch[i].numpy().shape)
        print("label: ", label)
        plt.imshow(images_batch[i].numpy().astype("uint8"))
        plt.title(decode_one_hot(label))
    plt.show()

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

def train_model(model, training_data, validation_data, number_of_epochs):
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

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=10)

    model.fit(
    training_data,
    validation_data=validation_data,
    epochs=number_of_epochs,
    callbacks=[tensorboard_callback])

    return model

def test(model, test_data):
    yhat = model.predict(test_data)
    yhat = yhat.round()
    y_test = np.concatenate([y for x, y in test_data], axis=0)
    report = classification_report(y_test, yhat,target_names=['N','D','G','C','A','H','M','O'],output_dict=True)
    df = pd.DataFrame(report).transpose()
    print(df)


def load_datasets():
    global label_dict
    df = pd.read_csv('./input/ocular-disease-recognition-odir5k/full_df.csv')
    label_dict = build_label_dictionary(df)


    train, val, test = load_sample_ids(df, 500)

    training_data = configure_for_performance(prepare_data(train))
    validation_data = configure_for_performance(prepare_data(val))
    test_data = configure_for_performance(prepare_data(test))

    return training_data, validation_data, test_data

def main():
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--show', action='store_true', help='Visualize a training batch')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--test', action='store_true', help='Test model')
    parser.add_argument('--dump', action='store_true', help='Dump data from first examples')
    parser.add_argument('--name', type=str, help='Name of the model', default="tmpModel")
    parser.add_argument('--epochs', type=int, help='Number of epochs to train', default=40)
    args = parser.parse_args()


    training_data, validation_data, test_data = load_datasets()

    if args.show:
        show_batch(training_data)
    
    if args.dump:
        print_dataset_stats(["training_data"],[training_data],5)

    if args.train:
        trained_model = train_model(create_model(), training_data, validation_data, args.epochs)
        trained_model.save('models/' + args.name)

    if args.test:
        model = tf.keras.models.load_model('models/' + args.name)
        test(model, test_data)


if __name__ == '__main__':
    main()