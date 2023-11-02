# Fabric Defect Classification with CNNs

## Overview
This Python script is designed for fabric defect classification using Convolutional Neural Networks (CNNs). The code utilizes the TensorFlow and Keras libraries for building and training CNN models. The dataset consists of fabric images with five categories of defects: Hole, Horizontal Line, Isolated Defect, Needle Lines, and Vertical Line.

## Prerequisites
Make sure you have the following libraries installed:

TensorFlow
Keras
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
You can install these dependencies using the following command:

!pip install tensorflow keras pandas numpy matplotlib seaborn scikit-learn

## Usage
1. Clone the repository or download the script.

2. Ensure your dataset is organized in a folder structure compatible with Keras' flow_from_directory function. The dataset path and categories should be set in the script:

dataset_path = r"your_dataset_path"
categories = ["Hole", "Horizontal Line", "Isolated Defect", "Needle Lines", "Vertical Line"]

3. Adjust the data augmentation and preprocessing settings as needed in the ImageDataGenerator section:

data_generator = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    channel_shift_range=0.2,
    fill_mode='nearest'
)

4. Choose the desired model architecture by uncommenting the relevant lines. Options include a custom CNN, ResNet50, and InceptionV3.

5. Train and evaluate the model by running the script.

## Project Structure
- `fabric_defect_classification.py`: Main script
- `model_architectures.py`: Model architecture definitions
- `data_preprocessing.py`: Data preprocessing functions
- `evaluation_and_visualization.py`: Evaluation and visualization functions
- `README.md`: Project documentation