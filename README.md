# deeplearning-skincancer2021-udec
Analisis of Skin Cancer MNIST: HAM10000 dataset using state-of-the-art deep learning methods

## Authors
- Ricardo Ávila Crisóstomo
- Heraldo Hernandez
- Carlos Farkas

## Links

#### Dataset link:
- https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
#### Data description: 
- https://arxiv.org/ftp/arxiv/papers/1803/1803.10417.pdf

#### Papers
- https://www.thelancet.com/journals/lanonc/article/PIIS1470-2045(19)30333-X/fulltext
- https://www.nature.com/articles/s41591-020-0942-0

#### Data download links:
```
- https://usegalaxy.org/datasets/bbd44e69cb8906b56354443fbb98d9c1/display?to_ext=zip  # images part 1
- https://usegalaxy.org/datasets/bbd44e69cb8906b5a14a2bc8e9f56106/display?to_ext=zip  # images part 2
- https://usegalaxy.org/datasets/bbd44e69cb8906b51cd45901a7e94f25/display?to_ext=csv  # metadata
```
Ubuntu/Unix: To create and download images into "HAM10000" folder, do the following:
```
mkdir HAM10000 && cd HAM10000
wget -O HAM10000_images_part_1.zip https://usegalaxy.org/datasets/bbd44e69cb8906b56354443fbb98d9c1/display?to_ext=zip
wget -O HAM10000_images_part_2.zip https://usegalaxy.org/datasets/bbd44e69cb8906b5a14a2bc8e9f56106/display?to_ext=zip
wget -O HAM10000_metadata.csv https://usegalaxy.org/datasets/bbd44e69cb8906b51cd45901a7e94f25/display?to_ext=csv

mkdir HAM10000_images_part_1 HAM10000_images_part_2
unzip HAM10000_images_part_1.zip -d ./HAM10000_images_part_1/
unzip HAM10000_images_part_2.zip -d ./HAM10000_images_part_2/
```
After this, ~ 5000 images will be inflated in each directory (HAM10000_images_part_1 and HAM10000_images_part_2, respectively). 

## Python Library Requirements
```
### Tested on : Python 3.8.8 (default, Apr 13 2021, 15:08:03) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32

### Kernel install (Jupyter Notebook)
conda create -n ipykernel_py3 python=3 ipykernel
source conda activate ipykernel_py3    # On Windows, remove the word 'source'
python -m ipykernel install --user

### Pandas: https://pandas.pydata.org/
pip install pandas

### Installing Tensorflow and keras
# Tensorflow: https://github.com/tensorflow/tensorflow
# Keras installation: https://keras.io/
pip install --upgrade pip
pip install --upgrade setuptools
pip install tensorflow
pip show tensorflow
pip install keras

# Checking libraries
pip list | grep tensorflow
pip list | grep keras

### scikit-learn: https://scikit-learn.org/stable/
pip install -U scikit-learn
python -m pip install -U scikit-image

### imageio: https://imageio.readthedocs.io/en/stable/installation.html
pip install imageio

### seaborn: https://seaborn.pydata.org/
pip install seaborn
```
## Images to h5 for python

## Usage
To implement the analysis, classic_CNN_block.py must be executed inside HAM10000 folder. The script will take as input the two folders containing ~5000 images each and the metadata (HAM10000_metadata.csv). In ubuntu 16.04: 

```
usage: classic_CNN_block.py [-h] [--size SIZE] [--epochs EPOCHS]
                            [--batch_size BATCH_SIZE] [--test_size TEST_SIZE]
                            [--rotation_range ROTATION_RANGE]
                            [--melanoma MELANOMA] [--aug_images AUG_IMAGES]
                            [--patience PATIENCE] [--min_delta MIN_DELTA]
                            [--metadata METADATA]

This script implements a regularized Convolutional Neural Network model (CNN)
on python to classify HAM10000 Images.

optional arguments:
  -h, --help            show this help message and exit
  --size SIZE           number of pixels to resize images (int). Default = 32
  --epochs EPOCHS       number of epochs (int). Default = 200
  --batch_size BATCH_SIZE
                        batch_size for batch_normalization (int). Default = 16
  --test_size TEST_SIZE
                        The proportion of the dataset to include in the test
                        split. If int, represents the absolute number of test
                        samples. Default = 0.17
  --rotation_range ROTATION_RANGE
                        rotation_range for Data Augmentation (degrees).
                        Default = 90
  --melanoma MELANOMA   class weight for melanoma. Higher than 1.0, the model
                        will be more sensitive to Melanoma (float). Default =
                        3.0
  --aug_images AUG_IMAGES
                        number of augmented images for training (int). Default
                        = 6000, type=int
  --patience PATIENCE   Number of epochs to wait before early stop if no
                        progress on the validation set. Default = 30, type=int
  --min_delta MIN_DELTA
                        Change in loss function to be considered as
                        improvement for EarlyStopping (float). Default =
                        0.0001, type=float
  --metadata METADATA   csv file containing HAM10000 metadata (str). Default =
                        HAM10000.csv
```
The script can be run as follows: 

```
python classic_CNN_block.py --size 32 --epochs 1000 --batch_size 50 --test_size 0.17 --rotation_range 180 --melanoma 3.0 --aug_images 3000 --patience 50 --min_delta 0.0001 --metadata HAM10000_metadata.csv
```
Setting  ```---melanoma 3.0 ```  will try to make the model more sensitive to melanoma

## TO-DO
- [x] Add script parameters (sys.argv[0]) 
- [x] save/load models after training
- [x] Include data augmentation, early stopping, batch normalization and dropout
- [] Cargar modelos pre-entrenados (e.j. resnet18)
