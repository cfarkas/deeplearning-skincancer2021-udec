#!/usr/bin/env python
# coding: utf-8

import sys
import argparse
import itertools
import shutil

### Requirements
# pip install pandas
# pip install matplotlib
# pip install seaborn
# pip install -U scikit-learn
# pip3 install tensorflow

### Keras installation
# pip install --upgrade pip
# pip install --upgrade setuptools
# pip install tensorflow
# pip show tensorflow
# pip install keras

# pip list | grep tensorflow
# pip list | grep keras

parser = argparse.ArgumentParser(description="This script implements a regularized Convolutional Neural Network model (CNN) on python to classify HAM10000 Images.")
parser.add_argument('--size', help="number of pixels to resize images. Default = 32", type=int)
parser.add_argument('--epochs', help="number of epochs. Default = 50", type=int)
parser.add_argument('--batch_size', help="batch_size for batch_normalization. Default = 16", type=int)
parser.add_argument('--test_size', help="test_size, fraction of images selected for test. Default = 0.17", type=str)
parser.add_argument('--rotation_range', help="rotation_range for Data Augmentation (degrees). Default = 90", type=int)
args = parser.parse_args()

name = sys.argv[0]
size = int(sys.argv[2])
EPOCHS = int(sys.argv[4])
BATCH_SIZE = int(sys.argv[6])
TEST_SIZE = float(sys.argv[8])
ROTATION_RANGE = int(sys.argv[10])

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    OKRED = '\033[91m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print(bcolors.OKGREEN + "Command line:", str(sys.argv) + bcolors.ENDC)
print("")
print(bcolors.OKGREEN + "--- Loading python libraries --" + bcolors.ENDC)
print("")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
import imageio
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder

print(bcolors.OKGREEN + " -- Setting 42 as Random Seed Number --" + bcolors.ENDC)
print("")
np.random.seed(42)

print(bcolors.OKGREEN + "--- The number of pixels to resize images will be: ---" + bcolors.ENDC)
print("")
SIZE=size
print(SIZE)
print("")

print(bcolors.OKGREEN + "--- Creating training and validation directories and importing metadata ---" + bcolors.ENDC)
print("")

path = os.getcwd()
data_dir = os.listdir(path)
metadata_df = pd.read_csv('HAM10000_metadata.csv')

# Create a new directory
base_dir = 'base_dir'
os.mkdir(base_dir)

# create a path to 'base_dir' to which we will join the names of the new folders
# train_dir
train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)

# val_dir
val_dir = os.path.join(base_dir, 'val_dir')
os.mkdir(val_dir)

# create new folders inside train_dir
nv = os.path.join(train_dir, 'nv')
os.mkdir(nv)
mel = os.path.join(train_dir, 'mel')
os.mkdir(mel)
bkl = os.path.join(train_dir, 'bkl')
os.mkdir(bkl)
bcc = os.path.join(train_dir, 'bcc')
os.mkdir(bcc)
akiec = os.path.join(train_dir, 'akiec')
os.mkdir(akiec)
vasc = os.path.join(train_dir, 'vasc')
os.mkdir(vasc)
df = os.path.join(train_dir, 'df')
os.mkdir(df)

# create new folders inside val_dir
nv = os.path.join(val_dir, 'nv')
os.mkdir(nv)
mel = os.path.join(val_dir, 'mel')
os.mkdir(mel)
bkl = os.path.join(val_dir, 'bkl')
os.mkdir(bkl)
bcc = os.path.join(val_dir, 'bcc')
os.mkdir(bcc)
akiec = os.path.join(val_dir, 'akiec')
os.mkdir(akiec)
vasc = os.path.join(val_dir, 'vasc')
os.mkdir(vasc)
df = os.path.join(val_dir, 'df')
os.mkdir(df)

print(bcolors.OKGREEN + "--- Loading a sample of the metadata ---" + bcolors.ENDC)
print("")
metadata_df.sample(2)

print(bcolors.OKGREEN + "--- Printing classes names ---" + bcolors.ENDC)
print("")
classes_name = list(metadata_df['dx'].unique())
print(classes_name)
print("")

print(bcolors.OKGREEN + "--- Codifiying lesion types as numbers with LabelEncoder ---" + bcolors.ENDC)
print("")
le = LabelEncoder()
le.fit(metadata_df['dx'])
LabelEncoder()
print(list(le.classes_))
metadata_df['label'] = le.transform(metadata_df["dx"])
print(metadata_df.sample(10))

print(bcolors.OKGREEN + "--- Taking a close look of metadata ---" + bcolors.ENDC)
print("")
lesion_counts = metadata_df['dx'].value_counts() # counting occurrences by class
lesion_counts = lesion_counts.to_frame()      # pandas series core to dataframe
lesion_counts

print(bcolors.OKRED + "--- Plotting Skin Lesion counts by type---" + bcolors.ENDC)
print("")
plt_a = lesion_counts.plot(kind='bar', color = ['darkred'], figsize=(6,4.5)) # , color=['darkblue']
plt_a.set_ylabel('Counts', fontsize=14)
plt_a.set_xlabel('Lesion', fontsize=14)
plt_a.set_title('Type of lesión', fontsize=15)
plt_a.tick_params(axis='both', which='major', labelsize=13)

plt_a.axhline(y=500, color='gray', linestyle='--')
plt_a.axhline(y=1000,color='gray', linestyle='--')
plt_a.text(6.9, 1000, '1000', fontsize=10, va='center', ha='center', backgroundcolor='w')
plt_a.text(6.85, 500, '500', fontsize=10, va='center', ha='center', backgroundcolor='w')
plt_a.get_legend().remove()
plt_a.figure.savefig('lesion_counts.png', bbox_inches = 'tight')
plt.close()

sex_counts = metadata_df['sex'].value_counts() # counting occurrences by class
sex_counts = sex_counts.to_frame()           # pandas series core to dataframe
sex_counts

print(bcolors.OKRED + "--- Plotting Skin Lesion counts by sex ---" + bcolors.ENDC)
print("")
plt_b = sex_counts.plot(kind='bar', color = ['darkred'], figsize=(3,4.5)) # , color=['darkblue']
plt_b.set_ylabel('Counts', fontsize=14)
plt_b.set_xlabel('Sex', fontsize=14)
plt_b.set_title('Lesions by sex', fontsize=15)
plt_b.tick_params(axis='both', which='major', labelsize=13)
plt_b.get_legend().remove()
plt_b.figure.savefig('sex_counts.png', bbox_inches = 'tight')
plt.close()

localization_counts = metadata_df['localization'].value_counts() # counting occurrences by class
localization_counts = localization_counts.to_frame()           # pandas series core to dataframe
localization_counts

print(bcolors.OKRED + "--- Plotting Skin Lesion counts by localization ---" + bcolors.ENDC)
print("")
plt_c = localization_counts.plot(kind='bar', color = ['darkred'], figsize=(8,5)) # , color=['darkblue']
plt_c.set_ylabel('Counts', fontsize=14)
plt_c.set_xlabel('Localization', fontsize=14)
plt_c.set_title('Lesions by localization', fontsize=15)
plt_c.tick_params(axis='both', which='major', labelsize=13)
plt_c.axhline(y=300, color='gray', linestyle='--')
plt_c.axhline(y=100,color='gray', linestyle='--')
plt_c.text(15, 300, '300', fontsize=10, va='center', ha='center', backgroundcolor='w')
plt_c.text(15, 100, '100', fontsize=10, va='center', ha='center', backgroundcolor='w')
plt_c.get_legend().remove()
plt_c.figure.savefig('localization_counts.png', bbox_inches = 'tight')
plt.close()

expert_validation_counts = metadata_df['dx_type'].value_counts()      # counting occurrences by class
expert_validation_counts = expert_validation_counts.to_frame()      # pandas series core to dataframe
expert_validation_counts

print(bcolors.OKRED + "--- Plotting Skin Lesion expert validation ---" + bcolors.ENDC)
print("")
plt_d = expert_validation_counts.plot(kind='bar', color = ['darkred'], figsize=(3.5,4.5)) # , color=['darkblue']
plt_d.set_ylabel('Counts', fontsize=14)
plt_d.set_xlabel('Class', fontsize=14)
plt_d.set_title('Expert knowledge of images', fontsize=15)
plt_d.tick_params(axis='both', which='major', labelsize=13)
plt_d.axhline(y=1000, color='gray', linestyle='--')
plt_d.text(3.9, 1000, '1000', fontsize=10, va='center', ha='center', backgroundcolor='w')
plt_d.get_legend().remove()
plt_d.figure.savefig('expert_validation_counts.png', bbox_inches = 'tight')
plt.close()

sample_age = metadata_df[pd.notnull(metadata_df['age'])]
a = sns.distplot(sample_age['age'], fit=stats.norm, color='red');
a.axes.set_title("Distribución de Edad",fontsize=15)
a.set_xlabel("edad",fontsize=14)
a.set_ylabel("Densidad",fontsize=14)

print(bcolors.OKRED + "--- Plotting Skin Lesion by age ---" + bcolors.ENDC)
print("")
sns.set(rc={'figure.figsize':(8,4)})
a.figure.savefig('sample_age.png', bbox_inches = 'tight')
plt.close()

print(bcolors.OKGREEN + "--- Data by class ---" + bcolors.ENDC)
print("")
from sklearn.utils import resample
print(metadata_df['label'].value_counts())
print("")


print(bcolors.OKGREEN + "--- Creating a stratified validation set ---" + bcolors.ENDC)
print("")

# this will tell us how many images are associated with each lesion_id
df = metadata_df.groupby('lesion_id').count()

# now we filter out lesion_id's that have only one image associated with it
df = df[df['image_id'] == 1]
df.reset_index(inplace=True)
df.head()


print(bcolors.OKGREEN + "--- Detecting lesions with duplicate images ---" + bcolors.ENDC)
print("")

# here we identify lesion_id's that have duplicate images and those that have only
# one image.

def identify_duplicates(x):
    unique_list = list(df['lesion_id'])
    if x in unique_list:
        return 'no_duplicates'
    else:
        return 'has_duplicates'
# create a new colum that is a copy of the lesion_id column
metadata_df['duplicates'] = metadata_df['lesion_id']
# apply the function to this new column
metadata_df['duplicates'] = metadata_df['duplicates'].apply(identify_duplicates)
metadata_df.head()

print(bcolors.OKGREEN + "--- Filtering out duplicates ---" + bcolors.ENDC)
print("")

metadata_df['duplicates'].value_counts()
# now we filter out images that don't have duplicates
df = metadata_df[metadata_df['duplicates'] == 'no_duplicates']
df.shape

print(bcolors.OKGREEN + "--- Creating a validation set without duplicates ---" + bcolors.ENDC)
print("")

# now we create a val set using df because we are sure that none of these images
# have augmented duplicates in the train set

y = df['dx']
_, df_val = train_test_split(df, test_size=TEST_SIZE, random_state=42, stratify=y)
df_val.shape
df_val['dx'].value_counts()


print(bcolors.OKGREEN + "--- Creating a training set that excludes images that are in the validation set ---" + bcolors.ENDC)
print("")

# This set will be df_data excluding all rows that are in the val set
# This function identifies if an image is part of the train
# or val set.
def identify_val_rows(x):
    # create a list of all the lesion_id's in the val set
    val_list = list(df_val['image_id'])
    if str(x) in val_list:
        return 'val'
    else:
        return 'train'

# identify train and val rows
# create a new colum that is a copy of the image_id column
metadata_df['train_or_val'] = metadata_df['image_id']

# apply the function to this new column
metadata_df['train_or_val'] = metadata_df['train_or_val'].apply(identify_val_rows)

# filter out train rows
df_train = metadata_df[metadata_df['train_or_val'] == 'train']


print(bcolors.OKGREEN + "--- Training set length: ---" + bcolors.ENDC)
print(len(df_train))
print("")
print(df_train['dx'].value_counts())
print("")

print(bcolors.OKGREEN + "--- Validation set length: ---" + bcolors.ENDC)
print(len(df_val))
print("")
print(df_val['dx'].value_counts())
print("")


print(bcolors.OKGREEN + "--- Transfering the Images into the training and validation folders: ---" + bcolors.ENDC)
print("")
# Set the image_id as the index in df_data
metadata_df.set_index('image_id', inplace=True)

# Get a list of images in each of the two folders
folder_1 = os.listdir('./HAM10000_images_part_1')
folder_2 = os.listdir('./HAM10000_images_part_2')

# Get a list of train and val images
train_list = list(df_train['image_id'])
val_list = list(df_val['image_id'])

# Transfer the train images

for image in train_list:

    fname = image + '.jpg'
    label = metadata_df.loc[image,'dx']

    if fname in folder_1:
        # source path to image
        src = os.path.join('./HAM10000_images_part_1', fname)
        # destination path to image
        dst = os.path.join(train_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    if fname in folder_2:
        # source path to image
        src = os.path.join('./HAM10000_images_part_2', fname)
        # destination path to image
        dst = os.path.join(train_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

# Transfer the val images

for image in val_list:

    fname = image + '.jpg'
    label = metadata_df.loc[image,'dx']

    if fname in folder_1:
        # source path to image
        src = os.path.join('./HAM10000_images_part_1', fname)
        # destination path to image
        dst = os.path.join(val_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    if fname in folder_2:
        # source path to image
        src = os.path.join('./HAM10000_images_part_2', fname)
        # destination path to image
        dst = os.path.join(val_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

# check how many val images we have in each folder

print(bcolors.OKCYAN + "--- Number of Nevus images in validation directory: ---" + bcolors.ENDC)
print(len(os.listdir('base_dir/val_dir/nv')))
print("")
print(bcolors.OKCYAN + "--- Number of Melanoma images in validation directory: ---" + bcolors.ENDC)
print(len(os.listdir('base_dir/val_dir/mel')))
print("")
print(bcolors.OKCYAN + "--- Number of Benign keratosis (bkl) images in validation directory: ---" + bcolors.ENDC)
print(len(os.listdir('base_dir/val_dir/bkl')))
print("")
print(bcolors.OKCYAN + "--- Number of Basal Cell Carcinoma (bcc) images in validation directory: ---" + bcolors.ENDC)
print(len(os.listdir('base_dir/val_dir/bcc')))
print("")
print(bcolors.OKCYAN + "--- Number of Actinic Keratoses (Solar Keratoses) and intraepithelial Carcinoma (Bowen’s disease) (akiec) images in validation directory: ---" + bcolors.ENDC)
print(len(os.listdir('base_dir/val_dir/akiec')))
print("")
print(bcolors.OKCYAN + "--- Number of Vascular skin lesions (vasc) images in validation directory: ---" + bcolors.ENDC)
print(len(os.listdir('base_dir/val_dir/vasc')))
print("")
print(bcolors.OKCYAN + "--- Number of Dermatofibroma (df) images in validation directory: ---" + bcolors.ENDC)
print(len(os.listdir('base_dir/val_dir/df')))
print("")

print(bcolors.OKGREEN + "--- Copying the training images into the augmentation directory and perform Data Augmentation---" + bcolors.ENDC)
print("")
print(bcolors.OKGREEN + "--- Using the provided size for Data Augmentation---" + bcolors.ENDC)
print("")
print(bcolors.OKGREEN + "--- We will not augment the nevus class ---" + bcolors.ENDC)
print("")

# note that we are not augmenting class 'nv'
class_list = ['mel','bkl','bcc','akiec','vasc','df']

for item in class_list:
    # We are creating temporary directories here because we delete these directories later
    # create a base dir
    aug_dir = 'aug_dir'
    os.mkdir(aug_dir)
    # create a dir within the base dir to store images of the same class
    img_dir = os.path.join(aug_dir, 'img_dir')
    os.mkdir(img_dir)
    # Choose a class
    img_class = item

    # list all images in that directory
    img_list = os.listdir('base_dir/train_dir/' + img_class)

    # Copy images from the class train dir to the img_dir e.g. class 'mel'
    for fname in img_list:
            # source path to image
            src = os.path.join('base_dir/train_dir/' + img_class, fname)
            # destination path to image
            dst = os.path.join(img_dir, fname)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)

    # point to a dir containing the images and not to the images themselves
    path = aug_dir
    save_path = 'base_dir/train_dir/' + img_class

    # Create a data generator
    datagen = ImageDataGenerator(
        rotation_range=ROTATION_RANGE,
        width_shift_range=0.1,                      # add to script parameters ?
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        #brightness_range=(0.9,1.1),
        fill_mode='nearest')

    aug_datagen = datagen.flow_from_directory(path,
                                           save_to_dir=save_path,
                                           save_format='jpg',
                                                    target_size=(size,size),
                                                    batch_size=BATCH_SIZE)

  # Generate the augmented images and add them to the training folders

    ###########

    num_aug_images_wanted = 6000 # total number of images we want to have in each class

    ###########

    num_files = len(os.listdir(img_dir))
    num_batches = int(np.ceil((num_aug_images_wanted-num_files)/BATCH_SIZE))

    # run the generator and create about 6000 augmented images
    for i in range(0,num_batches):

        imgs, labels = next(aug_datagen)

    # delete temporary directory with the raw image files
    shutil.rmtree('aug_dir')

# Check how many train images we now have in each folder.
# This is the original images plus the augmented images.
print(bcolors.OKCYAN + "--- Number of Nevus images in training directory (original + augmented images): ---" + bcolors.ENDC)
print(len(os.listdir('base_dir/train_dir/nv')))
print("")
print(bcolors.OKCYAN + "--- Number of Melanoma images in training directory (original + augmented images): ---" + bcolors.ENDC)
print(len(os.listdir('base_dir/train_dir/mel')))
print("")
print(bcolors.OKCYAN + "--- Number of Benign keratosis (bkl) images in training directory (original + augmented images): ---" + bcolors.ENDC)
print(len(os.listdir('base_dir/train_dir/bkl')))
print("")
print(bcolors.OKCYAN + "--- Number of Basal Cell Carcinoma (bcc) images in training directory (original + augmented images): ---" + bcolors.ENDC)
print(len(os.listdir('base_dir/train_dir/bcc')))
print("")
print(bcolors.OKCYAN + "--- Number of Actinic Keratoses (Solar Keratoses) and intraepithelial Carcinoma (Bowen’s disease) (akiec) images in training directory (original + augmented images): ---" + bcolors.ENDC)
print(len(os.listdir('base_dir/train_dir/akiec')))
print("")
print(bcolors.OKCYAN + "--- Number of Vascular skin lesions (vasc) images in training directory (original + augmented images): ---" + bcolors.ENDC)
print(len(os.listdir('base_dir/train_dir/vasc')))
print("")
print(bcolors.OKCYAN + "--- Number of Dermatofibroma (df) images in training directory (original + augmented images): ---" + bcolors.ENDC)
print(len(os.listdir('base_dir/train_dir/df')))
print("")

print(bcolors.OKGREEN + "--- Setting up the Generators: ---" + bcolors.ENDC)
print("")
train_path = 'base_dir/train_dir'
valid_path = 'base_dir/val_dir'

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = BATCH_SIZE
val_batch_size = BATCH_SIZE
image_size = size
train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

#datagen = ImageDataGenerator(
#    preprocessing_function= \
#    tensorflow.keras.applications.mobilenet.preprocess_input)

train_batches = datagen.flow_from_directory(train_path,
                                            target_size=(image_size,image_size),
                                            batch_size=train_batch_size)

valid_batches = datagen.flow_from_directory(valid_path,
                                            target_size=(image_size,image_size),
                                            batch_size=val_batch_size)

# Note: shuffle=False causes the test dataset to not be shuffled
test_batches = datagen.flow_from_directory(valid_path,
                                            target_size=(image_size,image_size),
                                            batch_size=1,
                                            shuffle=False)


print(bcolors.OKGREEN + "--- Getting the labels that are associated with each index ---" + bcolors.ENDC)
print("")
# Get the labels that are associated with each index
print(valid_batches.class_indices)

print(bcolors.OKGREEN + "--- Adding weights to try to make the model more sensitive to melanoma ---" + bcolors.ENDC)
print("")
# Add weights to try to make the model more sensitive to melanoma

class_weights={
    0: 1.0, # akiec
    1: 1.0, # bcc
    2: 1.0, # bkl
    3: 1.0, # df
    4: 3.0, # mel # Try to make the model more sensitive to Melanoma.   PARAMETER??
    5: 1.0, # nv
    6: 1.0, # vasc
}

print(bcolors.OKCYAN + "--- Defining the training model ---" + bcolors.ENDC)
print("")
num_classes = 7

model = Sequential()
model.add(Conv2D(256, (3, 3), activation="relu", input_shape=(size, size, 3)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(32))
model.add(Dense(7, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])

print(bcolors.OKCYAN + "--- Training: ---" + bcolors.ENDC)
print("")
print(bcolors.OKCYAN + "--- We will employ Early Stopping and the best training models will be saved with a certain frequency---" + bcolors.ENDC)
print("")

import datetime

model_path = './weights/model.{epoch:02d}-{loss:.2f}.h5'                 # save time and val_loss as h5 file
keras_callbacks = [
      EarlyStopping(monitor='val_loss', patience=30, mode='min', min_delta=0.0001),  # comment this line to deactivate EarlyStopping
      ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)] # selecting the best weights according to the minimum of the cost function in the validation set

callback_list = [keras_callbacks]

batch_size = BATCH_SIZE
epochs = EPOCHS

history = model.fit(train_batches,
                              steps_per_epoch=train_steps,
                              class_weight=class_weights,
                              validation_data=valid_batches,
                              validation_steps=val_steps,
                              epochs=epochs,
                              verbose=2,
                              callbacks=callback_list)

score = model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])
print("")

score = model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])
print("")

print(bcolors.OKRED + "--- Plotting accuracy and loss on training and validation sets in each epoch ---" + bcolors.ENDC)
print("")

from matplotlib import pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt_e = plt.plot(epochs, loss, 'y', label='Training loss')
plt_e = plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt_e = plt.title('Training and validation loss')
plt_e = plt.xlabel('Epochs')
plt_e = plt.ylabel('Loss')
plt_e = plt.legend()
plt_e = plt.savefig('Training_and_validation_loss.png', bbox_inches = 'tight')
plt_e = plt.show()
plt.close()

acc = history.history['acc']
val_acc = history.history['val_acc']
plt_f = plt.plot(epochs, acc, 'y', label='Training acc')
plt_f = plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt_f = plt.title('Training and validation accuracy')
plt_f = plt.xlabel('Epochs')
plt_f = plt.ylabel('Accuracy')
plt_f = plt.legend()
plt_f = plt.savefig('Training_and_validation_accuracy.png', bbox_inches = 'tight')
plt_f = plt.show()
plt.close()

print(bcolors.OKRED + "--- Saving training accuracy and loss curves ---" + bcolors.ENDC)
print("")

import numpy as np

train_loss = history.history['loss']
train_acc = history.history['acc']
val_loss = history.history['val_loss']
val_acc = history.history['val_acc']

np.save('train_loss.npy', train_loss)
np.save('train_acc.npy', train_acc)
np.save('val_loss.npy', val_loss)
np.save('val_acc.npy', val_acc)

print(bcolors.OKGREEN + "--- Prediction on test data ---" + bcolors.ENDC)
print("")
y_pred = model.predict(X_test)

print(bcolors.OKGREEN + "--- Convert predictions classes to one hot vectors ---" + bcolors.ENDC)
print("")
y_pred_classes = np.argmax(y_pred, axis = 1)

print(bcolors.OKGREEN + "--- Convert test data to one hot vectors ---" + bcolors.ENDC)
print("")
y_true = np.argmax(y_test, axis = 1)

y_pred_classes

print(bcolors.OKGREEN + "--- Obtaining Confusion matrix ---" + bcolors.ENDC)
print("")
cm = confusion_matrix(y_true, y_pred_classes)
fig, ax = plt.subplots(figsize=(6,6))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)
fig.savefig('confusion_matrix.png', bbox_inches = 'tight')
plt.close()

print(bcolors.OKRED + "--- Plotting fractional incorrect misclassifications ---" + bcolors.ENDC)
print("")
incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
plt.bar(np.arange(7), incorr_fraction)
plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')
plt.savefig('incorrect_misclassifications.png', bbox_inches = 'tight')
plt.close()
print(bcolors.OKGREEN + "##################" + bcolors.ENDC)
print(bcolors.OKGREEN + " --- All done ---" + bcolors.ENDC)
print(bcolors.OKGREEN + "##################" + bcolors.ENDC)
