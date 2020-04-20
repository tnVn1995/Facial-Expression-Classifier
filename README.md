# Facial Expression Classifier

Building an emotion classifier using tensorflow-gpu 2.0 and Opencv4.

## Data

Data is taken from Kaggle [fer2013](https://www.kaggle.com/deadskull7/fer2013). 
The dataset contains 34034 images of 7 basic emotions stored in a csv file.
Each image is of size 48x48.

## Requirement 

1. [Anaconda](https://www.anaconda.com/distribution/#download-section)
2. Tensorflow-gpu2 (This can be automatically [set up](https://anaconda.org/anaconda/tensorflow-gpu) 
using Anaconda)
3. Opencv4 (Also can be [set up](https://anaconda.org/conda-forge/opencv) using anaconda)

## Usage

1. Clone the Repo to your local machine
2. Move to the directory that contains the cloned repo

*Optional:* if you want to rerun the notebook:
3. Download fer2013 data and save it to folder data

4. Install the virtual environment using conda from the environment.yml file
> conda env create -f environment.yml
5. Once installed, activate the environment
> conda activate tf-gpu2
6. Type in command line the following command:
> python recognize_video.py -d face_detection_model -pre models/bn_dp_image_augmentation/bn_dp_augmentation.h5