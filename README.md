# deeplearning-skincancer2021-udec
Analisis of Skin Cancer MNIST: HAM10000 dataset using state-of-the-art deep learning methods

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
After this, 10015 images will be inflated in the HAM10000_images_part_1 and HAM10000_images_part_2 directories. 

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

## Execution
