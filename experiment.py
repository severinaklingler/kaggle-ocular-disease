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

tf.config.run_functions_eagerly(True)

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 16
img_height = 224
img_width = 224
final_class = 8
image_channels = 3
class_names = [
    "Normal (N)",
    "Diabetes (D)",
    "Glaucoma (G)",
    "Cataract (C)",
    "Age related Macular Degeneration (A)",
    "Hypertension (H)",
    "Pathological Myopia (M)",
    "Other diseases/abnormalities (O)"
]

def file_exists(file_path):
    [exists] = tf.py_function(_file_exists, [file_path], [tf.bool])
    exists.set_shape([])

    return exists

def _file_exists(file_path):
    return tf.io.gfile.exists(file_path.numpy())

def build_label_dictionary(df):
    dict = {}
    for index, row in df.iterrows():
        image_id = int(row['ID'])
        image_target = np.asarray(eval(row["target"]))
        dict[image_id] = image_target
    return dict

data_dir = pathlib.Path("./input/ocular-disease-recognition-odir5k/preprocessed_images/")
df = pd.read_csv('./input/ocular-disease-recognition-odir5k/full_df.csv')
label_dict = build_label_dictionary(df)

def create_model():
    inp1 = Input(shape=(img_height,img_width,image_channels), name="left")
    inp2 = Input(shape=(img_height,img_width,image_channels), name="right")
    new_input = Input(shape=(img_height,img_width, image_channels), name="New Input")

    conv1 = Conv2D(3, kernel_size=3, padding ='same', activation='relu', name="conleft1")(inp1)
    i1 = tf.keras.applications.ResNet50(include_top=False,weights="imagenet",input_tensor=new_input,input_shape=None, pooling='avg')(inp1)

    conv2= Conv2D(3, kernel_size=3, padding ='same', activation='relu', name="conright1")(inp2)
    i2 = tf.keras.applications.ResNet50V2(include_top=False,weights="imagenet",input_tensor=new_input,input_shape=None,pooling='avg')(inp2)


    merge = concatenate([i1,i2])
    class1 = Dense(1024, activation='relu')(merge)
    # class1 = Dense(512, activation='relu')(class1)
    class1 = Dense(256, activation='relu')(class1)
    # class1 = Dense(128, activation='relu')(class1)
    class1 = Dense(64, activation='relu')(class1)
    output = Dense(final_class, activation='sigmoid')(class1)
    model = Model(inputs=[inp1,inp2], outputs=output)
    return model


def _extract_label(part):
    one_hot = []
    match = re.search("\d+", str(part))
    if match: 
        image_id = int(match.group())
        one_hot = label_dict[image_id]
    else:
        print("no match found")
    return one_hot

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    [label] = tf.py_function(_extract_label, [parts[-1]], [tf.int64])
    label.set_shape([8])
    return label

def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=image_channels)
    return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    right_image = tf.io.read_file(file_path)
    right_image = decode_img(right_image)   

    left_image = right_image
    left_path = tf.strings.regex_replace(file_path, "right", "left", replace_global=True, name=None)
    if file_exists(left_path):
        left_image = tf.io.read_file(left_path)
        left_image = decode_img(left_image)
    
    return {"left" : left_image, "right":  right_image}, label


image_filenames = list(data_dir.glob('*.jpg'))
image_count = len(image_filenames)

list_ds = tf.data.Dataset.list_files(str(data_dir/'*_right.jpg'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)



for f in list_ds.take(5):
  print(f.numpy())
  print(f"Label : {get_label(f)}")

val_size = 500
val_ds = list_ds.take(val_size)
test_ds = list_ds.skip(val_size).take(val_size)
train_ds = list_ds.skip(2*val_size)

print(f"Training size: {tf.data.experimental.cardinality(train_ds).numpy()}")
print(f"Validation size: {tf.data.experimental.cardinality(val_ds).numpy()}")
print(f"Validation size: {tf.data.experimental.cardinality(test_ds).numpy()}")

train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for images, label in train_ds.take(10):
  print("Image shape: ", images["left"].numpy().shape)
  print("Label: ", label.numpy())


def configure_for_performance(ds):
#   ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds)

# images_batch, label_batch = next(iter(train_ds))

# plt.figure(figsize=(10, 10))
# for i in range(8):
#   ax = plt.subplot(2, 4, i + 1)
#   label = label_batch[i]
#   print("Image shape: ", images_batch["left"][i].numpy().shape)
#   print("label: ", label)
#   plt.imshow(images_batch["left"][i].numpy().astype("uint8"))
#   plt.title(class_names[label])
#   plt.imshow(images_batch["right"][i].numpy().astype("uint8"))
#   plt.title(class_names[label])
#   plt.axis("off")

# plt.show()


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
  train_ds,
  validation_data=val_ds,
  epochs=50
)


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
yhat = model.predict(test_ds)
yhat = yhat.round()
y_test = np.concatenate([y for x, y in test_ds], axis=0)
report = classification_report(y_test, yhat,target_names=['N','D','G','C','A','H','M','O'],output_dict=True)
df = pd.DataFrame(report).transpose()
print(df)