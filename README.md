# Street View Home Number Detection & Recognition

This project use Format 1 dataset from **SVHN dataset**, obtained from house numbers in Google Street View images.  

These are the original, variable-resolution, color house-number images with character level bounding boxes, as shown in the examples images above. (The blue bounding boxes in the example image are just for illustration purposes. The bounding box information are stored in digitStruct.mat instead of drawn directly on the images in the dataset.)  

Each tar.gz file contains the orignal images in png format, together with a **digitStruct.mat** file, which can be loaded using Matlab. The **digitStruct.mat** file contains a struct called digitStruct with the same length as the number of original images. Each element in **digitStruct** has the following fields: **name** which is a string containing the filename of the corresponding image, **bbox** which is a struct array that contains the position, size and label of each digit bounding box in the image.  

Eg: digitStruct(300).bbox(2).height gives height of the 2nd digit bounding box in the 300th image.  

You can download this dataset [here](http://ufldl.stanford.edu/housenumbers/).

## How to use this project  

1. Download dataset from this link [here](http://ufldl.stanford.edu/housenumbers/).
2. Extract training and testing dataset to folder name **data**
3. Clone this Git repository, you'll have a folder name **cv-ndetect** contain all the source code. Your working folder now would look like this:
```
.
|-- cv-ndetect/
|   |-- 16x32/
|-- data
    |-- test/
    |-- train/
```
You can delete all the *.mat* and *.m* in folder **test** and **train** or moving them to another folder. Those files contain bounding boxes of numbers in image and will be replaced by 2 csv files ***train_label.csv*** and ***test_label.csv***.  

**cv-ndetect** folder contains the following files:
  + **hn_detect.py**: All functions used in Jupyter notebook files are declared in this file.  
  + **train_label.csv, test_label.csv**: Training set label and Testing set label rewritten in csv form. Each row contain: *image name*, *top, left* position of bounding box, *width, height* of bounding box
  + **convert_mat.m**: Matlab script used to load *.mat* file in training set and testing set and create *train_label.csv, test_label.csv* file
  + **pre_processing.ipynb**: Used to test pre-processing image function.
  
  Run the following Jupyter notebook files:
  + **data_inspection.ipynb**: Used to inspect training, testing images as well as training bounding box
  + **crop_by_label.ipynb**: Used to create True positive feature dataset for classifier model. This data set is named *HOG_16x32* and is saved in folder ***16x32***
  + **negative_dataset.ipynb**: Used to create True negative feature dataset for classifier model. This dataset is named *negative_set.p* and is saved in folder ***16x32***
  + **training.ipynb**: Used to load dataset, train classifier model and test model on test set. Trained models are saved in folder ***16x32***
