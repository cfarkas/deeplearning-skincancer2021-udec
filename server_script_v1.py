#!/usr/bin/env python
# coding: utf-8

#!/bin/bash

### Access
# ssh wslab@152.74.15.34

### Directory
# cd /home/wslab/HAM10000/

### Environment
# conda activate annotate_my_genomes


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

print("This script implements a regularized Convolutional Neural Network model in python to classify HAM10000 Images.")
print("This script requires two arguments.")
print("")
print("- size: set pixel size. Default = 32")
print("")
print("- epochs: set number of epochs. Default = 50")
print("")
print("Batch_size: batch_size for batch_normalization. Default = 16")
print("")

size = sys.argv[1]
EPOCHS = sys.argv[2]
BATCH_SIZE = sys.argv[3]

if(len(sys.argv) != 3):
    print("Failed to start the program, missing parameters. Check the description of the script at the beginning")
    exit(0)
else:

print(" -- Loading Libraries --")
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

import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder

print(" -- Setting 42 as Random Seed Number --")
np.random.seed(42)


print(" --- Importing metadata ---")
path = '/home/wslab/HAM10000/'
data_dir = os.listdir(path)
metadata = pd.read_csv('/home/wslab/HAM10000/HAM10000_metadata.csv')

print(" --- Setting size = 32 (resize images to 32 x 32 pixel size) --- ")
SIZE=size

print(" --- Codifiying lesion types as numbers with LabelEncoder --- ")
le = LabelEncoder()
le.fit(metadata['dx'])
LabelEncoder()
print(list(le.classes_))
metadata['label'] = le.transform(metadata["dx"]) 
print(metadata.sample(10))

print(" --- Taking a close look of metadata ---")

lesion_counts = metadata['dx'].value_counts() # contando las ocurrencias por clase
lesion_counts = lesion_counts.to_frame()     # pandas series core a dataframe
lesion_counts

print(" --- Plotting Skin Lesion counts by type---")
plt_a = lesion_counts.plot(kind='bar', color = ['darkred'], figsize=(6,4.5)) # , color=['darkblue']
plt_a.set_ylabel('Conteos', fontsize=14)
plt_a.set_xlabel('Lesión', fontsize=14)
plt_a.set_title('Tipo de lesión', fontsize=15)
plt_a.tick_params(axis='both', which='major', labelsize=13)

plt_a.axhline(y=500, color='gray', linestyle='--')
plt_a.axhline(y=1000,color='gray', linestyle='--')
plt_a.text(6.9, 1000, '1000', fontsize=10, va='center', ha='center', backgroundcolor='w')
plt_a.text(6.85, 500, '500', fontsize=10, va='center', ha='center', backgroundcolor='w')
plt_a.get_legend().remove()
plt_a.figure.savefig('lesion_counts.png', bbox_inches = 'tight')
plt.close()

sex_counts = metadata['sex'].value_counts() # contando las ocurrencias por clase
sex_counts = sex_counts.to_frame()     # pandas series core a dataframe
sex_counts

print(" --- Plotting Skin Lesion counts by sex ---")
plt_b = sex_counts.plot(kind='bar', color = ['darkred'], figsize=(3,4.5)) # , color=['darkblue']
plt_b.set_ylabel('Conteos', fontsize=14)
plt_b.set_xlabel('Sexo', fontsize=14)
plt_b.set_title('Lesiones por sexo', fontsize=15)
plt_b.tick_params(axis='both', which='major', labelsize=13)
plt_b.get_legend().remove()
plt_b.figure.savefig('sex_counts.png', bbox_inches = 'tight')
plt.close()

localization_counts = metadata['localization'].value_counts() # contando las ocurrencias por clase
localization_counts = localization_counts.to_frame()     # pandas series core a dataframe
localization_counts

print(" --- Plotting Skin Lesion counts by localization ---")
plt_c = localization_counts.plot(kind='bar', color = ['darkred'], figsize=(8,5)) # , color=['darkblue']
plt_c.set_ylabel('Conteos', fontsize=14)
plt_c.set_xlabel('Localización', fontsize=14)
plt_c.set_title('Lesiones por zona corporal', fontsize=15)
plt_c.tick_params(axis='both', which='major', labelsize=13)
plt_c.axhline(y=300, color='gray', linestyle='--')
plt_c.axhline(y=100,color='gray', linestyle='--')
plt_c.text(15, 300, '300', fontsize=10, va='center', ha='center', backgroundcolor='w')
plt_c.text(15, 100, '100', fontsize=10, va='center', ha='center', backgroundcolor='w')
plt_c.get_legend().remove()
plt_c.figure.savefig('localization_counts.png', bbox_inches = 'tight')
plt.close()

expert_validation_counts = metadata['dx_type'].value_counts()      # contando las ocurrencias por clase
expert_validation_counts = expert_validation_counts.to_frame()     # pandas series core a dataframe
expert_validation_counts

print(" --- Plotting Skin Lesion expert validation ---")
plt_d = expert_validation_counts.plot(kind='bar', color = ['darkred'], figsize=(3.5,4.5)) # , color=['darkblue']
plt_d.set_ylabel('Conteos', fontsize=14)
plt_d.set_xlabel('Clases', fontsize=14)
plt_d.set_title('Conocimiento Experto de las imágenes', fontsize=15)
plt_d.tick_params(axis='both', which='major', labelsize=13)
plt_d.axhline(y=1000, color='gray', linestyle='--')
plt_d.text(3.9, 1000, '1000', fontsize=10, va='center', ha='center', backgroundcolor='w')
plt_d.get_legend().remove()
plt_d.figure.savefig('expert_validation_counts.png', bbox_inches = 'tight')
plt.close()

sample_age = metadata[pd.notnull(metadata['age'])]
a = sns.distplot(sample_age['age'], fit=stats.norm, color='red');
a.axes.set_title("Distribución de Edad",fontsize=15)
a.set_xlabel("edad",fontsize=14)
a.set_ylabel("Densidad",fontsize=14)

print(" --- Plotting Skin Lesion by age ---")
sns.set(rc={'figure.figsize':(8,4)})
a.figure.savefig('sample_age.png', bbox_inches = 'tight')
plt.close()

print(" --- Resampling data by class ---") 
from sklearn.utils import resample
print(metadata['label'].value_counts())

print(" --- Balancing data by resampling 500 images ---")
# Muchas formas de equilibrar los datos ... también puede intentar asignar pesos durante model.fit
# Hay que separar cada clase, volver a muestrear y combinar en un solo dataframe

df_0 = metadata[metadata['label'] == 0]
df_1 = metadata[metadata['label'] == 1]
df_2 = metadata[metadata['label'] == 2]
df_3 = metadata[metadata['label'] == 3]
df_4 = metadata[metadata['label'] == 4]
df_5 = metadata[metadata['label'] == 5]
df_6 = metadata[metadata['label'] == 6]


# using resample
n_samples=500 
df_0_balanced = resample(df_0, replace=True, n_samples=n_samples, random_state=42) 
df_1_balanced = resample(df_1, replace=True, n_samples=n_samples, random_state=42) 
df_2_balanced = resample(df_2, replace=True, n_samples=n_samples, random_state=42)
df_3_balanced = resample(df_3, replace=True, n_samples=n_samples, random_state=42)
df_4_balanced = resample(df_4, replace=True, n_samples=n_samples, random_state=42)
df_5_balanced = resample(df_5, replace=True, n_samples=n_samples, random_state=42)
df_6_balanced = resample(df_6, replace=True, n_samples=n_samples, random_state=42)

print(" --- Combining balanced data into a new dataframe: skin_df_balanced ---")
skin_df_balanced = pd.concat([df_0_balanced, df_1_balanced, 
                              df_2_balanced, df_3_balanced, 
                              df_4_balanced, df_5_balanced, df_6_balanced])

print(" --- Checking distribution. At this point, all classes should be equilibrated ---")
print(skin_df_balanced['label'].value_counts())
skin_df_balanced

print(" --- Reading images based on image ID from CSV file ---")
print(" --- This ensures that the correct image is read for correct identification in the csv file---")
image_path = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('/home/wslab/HAM10000/', '*', '*.jpg'))}

image_path

print(" --- Defining images path and add them as a new column into the new dataframe ---")
skin_df_balanced['path'] = metadata['image_id'].map(image_path.get)
print(" --- Using the path to read the images ---")
skin_df_balanced['image'] = skin_df_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE,SIZE))))
skin_df_balanced.head(10)

print(" --- Converting the image column of the dataframe to a numpy array ---")
X = np.asarray(skin_df_balanced['image'].tolist())
X = X/255.  # Escalar valores a 0-1. También puede utilizar StandardScaler u otro metodo
Y=skin_df_balanced['label']  # Asignando valores de etiqueta a Y
Y_cat = to_categorical(Y, num_classes=7) # Convirtiendo variables a variables categóricas, ya que este es un problema de clasificación

print(" --- Splitting dataset into train and test groups, respectively ---")
x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)


print(" --- Defining the model ---")
print(" --- Users can apply autokeras to find the best model ---")
print(" --- Or load pre-trained networks such as mobilenet or VGG16 ---")

num_classes = 7

model = Sequential()
model.add(Conv2D(256, (3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)))
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

print(" --- Training. You can also use generator to use augmentation during training ---")

batch_size = BATCH_SIZE 
epochs = EPOCHS

history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size = batch_size,
    validation_data=(x_test, y_test),
    verbose=2)

score = model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])

print(" --- Plotting accuracy and loss on training and validation sets in each epoch ---")
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

print(" --- Prediction on test data ---")
y_pred = model.predict(x_test)

print(" --- Convert predictions classes to one hot vectors ---")
y_pred_classes = np.argmax(y_pred, axis = 1) 

print(" --- Convert test data to one hot vectors ---")
y_true = np.argmax(y_test, axis = 1) 

y_pred_classes

print(" --- Obtaining Confusion matrix ---")
cm = confusion_matrix(y_true, y_pred_classes)
fig, ax = plt.subplots(figsize=(6,6))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)
fig.savefig('confusion_matrix.png', bbox_inches = 'tight')
plt.close()

print(" --- Plotting fractional incorrect misclassifications ---")
incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
plt.bar(np.arange(7), incorr_fraction)
plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')
plt.savefig('incorrect_misclassifications.png', bbox_inches = 'tight')
plt.close()

print(" --- All done ---")
