import pickle

import skimage
from keras import layers

import os
import zipfile
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from skimage import exposure
from skimage.morphology import ball
from skimage.segmentation._felzenszwalb_cy import gaussian
from sklearn.metrics import accuracy_score, precision_score, recall_score


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

import time

from tensorflow import keras

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)


# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     tf.config.set_logical_device_configuration(
#         gpus[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=5292)]
#     )


# logical_gpus = tf.config.list_logical_devices('GPU')
# print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")

import nibabel as nib
from scipy import ndimage



def read_nifti_file(filepath):

    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):

    min = -1000
    max = 200
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


# def process_scan(path):
#     """Read and resize volume"""
#     # Read scan
#     volume = read_nifti_file(path)
#     # Normalize
#     volume = normalize(volume)
#     # Resize width, height and depth
#     volume = resize_volume(volume)
#     return volume


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    volume = normalize(volume)
    volume = exposure.equalize_hist(volume)
    volume = skimage.morphology.opening(volume, ball(2))
    volume = gaussian(volume, sigma=2)
    volume = resize_volume(volume)
    return volume


import gzip
import shutil


def unpack_single_gzip_in_folder(folder_path):

    files = os.listdir(folder_path)


    gz_files = [file for file in files if file.endswith('.gz')]


    if len(gz_files) == 1:
        gz_file_path = os.path.join(folder_path, gz_files[0])


        output_file_path = os.path.splitext(gz_file_path)[0]

        with gzip.open(gz_file_path, 'rb') as f_in, open(output_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        os.remove(gz_file_path)
        print(f"Распаковано: {gz_file_path} -> {output_file_path}")
    else:
        print("Ошибка: Не удалось определить единственный файл .gz в указанной папке.")


import dicom2nifti


# Folder "CT-0" consist of CT scans having normal lung tissue,
# no CT-signs of viral pneumonia.

# for x in os.listdir(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\700_800_studies"):
#     dicom2nifti.convert_directory(os.path.join(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\700_800_studies", x,
#                                                    os.listdir(os.path.join(
#                                                        r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\700_800_studies",
#                                                        x))[0]), os.path.join(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\700_800_studies", x))
#     unpack_single_gzip_in_folder(os.path.join(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\700_800_studies", x))




normal_scan_paths3 = [
    os.path.join(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\good\0_100_studies", x)
    for x in os.listdir(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\good\0_100_studies")
]

normal_scan_paths2 = [
    os.path.join(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\good\100_200_studies", x)
    for x in os.listdir(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\good\100_200_studies")
]

normal_scan_paths1 = [
    os.path.join(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\good\200_300_studies", x)
    for x in os.listdir(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\good\200_300_studies")
]

normal_scan_paths4 = [
    os.path.join(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\good\300_400_studies", x)
    for x in os.listdir(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\good\300_400_studies")
]

normal_scan_paths5 = [
    os.path.join(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\good\400_500_studies", x)
    for x in os.listdir(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\good\400_500_studies")
]

normal_scan_paths6 = normal_scan_paths1 + normal_scan_paths2 + normal_scan_paths3 + normal_scan_paths4 + normal_scan_paths5
normal_scan_paths = []

for dir in normal_scan_paths6:
    files = os.listdir(dir)
    nii_files = [file for file in files if file.endswith('.nii')]
    if len(nii_files) == 1:
        nii_file_path = os.path.join(dir, nii_files[0])
        normal_scan_paths.append(nii_file_path)

print(normal_scan_paths)


abnormal_scan_paths3 = [
    os.path.join(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\500_600_studies", x)
    for x in os.listdir(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\500_600_studies")
]

abnormal_scan_paths2 = [
    os.path.join(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\600_700_studies", x)
    for x in os.listdir(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\600_700_studies")
]

abnormal_scan_paths1 = [
    os.path.join(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\700_800_studies", x)
    for x in os.listdir(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\700_800_studies")
]

abnormal_scan_paths4 = [
    os.path.join(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\somebad", x)
    for x in os.listdir(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\somebad")
]

abnormal_scan_paths5 = abnormal_scan_paths1 + abnormal_scan_paths2 + abnormal_scan_paths3 + abnormal_scan_paths4
abnormal_scan_paths = []

for dir in abnormal_scan_paths5:
    files = os.listdir(dir)
    nii_files = [file for file in files if file.endswith('.nii')]
    if len(nii_files) == 1:
        nii_file_path = os.path.join(dir, nii_files[0])
        abnormal_scan_paths.append(nii_file_path)
#
print("CT scans with normal lung tissue: " + str(len(normal_scan_paths)))
print("CT scans with abnormal lung tissue: " + str(len(abnormal_scan_paths)))


normal_scan_paths = normal_scan_paths[:100]
abnormal_scan_paths = abnormal_scan_paths[:100]


start_time = time.time()
abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
normal_scans = np.array([process_scan(path) for path in normal_scan_paths])
end_time = time.time()
print(end_time-start_time)

state_to_save = {
    'abnormal_scans': abnormal_scans,
    'normal_scans': normal_scans,
}

with open('200saved_state333.pickle', 'wb') as file:
    pickle.dump(state_to_save, file)



# start_time = time.time()
# with open('saved_state700.pickle', 'rb') as file:
#     loaded_state = pickle.load(file)
#
# abnormal_scans = loaded_state['abnormal_scans']
# normal_scans = loaded_state['normal_scans']
# end_time = time.time()
# print(end_time-start_time)



abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])

# Split data in the ratio 70-30 for training and validation.
# x_train = np.concatenate((abnormal_scans[:63], abnormal_scans[190:253], normal_scans[100:226]), axis=0)
# y_train = np.concatenate((abnormal_labels[:63], abnormal_labels[190:253], normal_labels[100:226]), axis=0)
# x_val = np.concatenate((abnormal_scans[63:90], abnormal_scans[253:280], normal_scans[226:280]), axis=0)
# y_val = np.concatenate((abnormal_labels[63:90], abnormal_labels[253:280], normal_labels[226:280]), axis=0)
x_train = np.concatenate((abnormal_scans[:70], normal_scans[:70]), axis=0)
y_train = np.concatenate((abnormal_labels[:70], normal_labels[:70]), axis=0)
x_val = np.concatenate((abnormal_scans[70:], normal_scans[70:]), axis=0)
y_val = np.concatenate((abnormal_labels[70:], normal_labels[70:]), axis=0)
print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)


import random

from scipy import ndimage


@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):

    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):

    volume = tf.expand_dims(volume, axis=3)
    return volume, label



# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 1
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)


import matplotlib.pyplot as plt

data = train_dataset.take(1)
images, labels = list(data)[0]
images = images.numpy()
image = images[0]
print("Dimension of the CT scan is:", image.shape)
plt.imshow(np.squeeze(image[:, :, 30]), cmap="gray")



def plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of 20 CT slices"""
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


plot_slices(4, 10, 128, 128, image[:, :, :40])


def get_model(width=128, height=128, depth=64):

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="3dcnn")
    return model



model = get_model(width=128, height=128, depth=64)
model.summary()


initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)


checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification1111111.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)


epochs = 100
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)




fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

model.load_weights("3d_image_classification1111111.h5")
prediction = model.predict(np.expand_dims(x_val[0], axis=0))[0]
scores = [1 - prediction[0], prediction[0]]

class_names = ["normal", "abnormal"]
for score, name in zip(scores, class_names):
    print(
        "This model is %.2f percent confident that CT scan is %s"
        % ((100 * score), name)
    )

datagen = ImageDataGenerator()
generator = datagen.flow(x_val, batch_size=batch_size, shuffle=False)

y_pred = model.predict_generator(generator, steps=np.ceil(len(x_val) / batch_size))

y_pred_binary = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_val, y_pred_binary)
precision = precision_score(y_val, y_pred_binary)
recall = recall_score(y_val, y_pred_binary)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')


model.save('my_model1111111', save_format='tf')







def get_model2(width=128, height=128, depth=64):
    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("tanh")(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU(alpha=1.0)(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    improved_model = keras.Model(inputs, outputs, name="improved_3dcnn")
    return improved_model



model2 = get_model2(width=128, height=128, depth=64)
model2.summary()

"""
## Train model
"""


initial_learning_rate = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model2.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)


checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification222222222.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)


epochs = 200
model2.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)




fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model2.history.history[metric])
    ax[i].plot(model2.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

"""
## Make predictions on a single CT scan
"""


model2.load_weights("3d_image_classification222222222.h5")
prediction2 = model2.predict(np.expand_dims(x_val[0], axis=0))[0]
scores2 = [1 - prediction[0], prediction[0]]

class_names = ["normal", "abnormal"]
for score, name in zip(scores, class_names):
    print(
        "This model is %.2f percent confident that CT scan is %s"
        % ((100 * score), name)
    )



model2.load_weights("3d_image_classification222222222.h5")
prediction2 = model2.predict(np.expand_dims(x_val[0], axis=0))[0]
scores2 = [1 - prediction2[0], prediction2[0]]

class_names = ["normal", "abnormal"]
for score, name in zip(scores2, class_names):
    print(
        "This model is %.2f percent confident that CT scan is %s"
        % ((100 * score), name)
    )


datagen = ImageDataGenerator()
generator = datagen.flow(x_val, batch_size=batch_size, shuffle=False)

y_pred = model2.predict_generator(generator, steps=np.ceil(len(x_val) / batch_size))


y_pred_binary = (y_pred > 0.5).astype(int)


# метрики
accuracy = accuracy_score(y_val, y_pred_binary)
precision = precision_score(y_val, y_pred_binary)
recall = recall_score(y_val, y_pred_binary)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')


model2.save('my_model222222222', save_format='tf')