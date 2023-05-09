# Image Recognition System for Fruits 360 Dataset
This is a deep learning-based computer vision system that can classify different types of fruits in images using a convolutional neural network (CNN) trained on the "Fruits 360" dataset.

# Dataset
The "Fruits 360" dataset contains over 80,000 images of 120 different types of fruits. The dataset is divided into training, validation, and test sets, with 80% of the data used for training, 10% for validation, and 10% for testing.

# Preprocessing
The dataset images are resized to a fixed size of 224 x 224 pixels and normalized to the range [0, 1]. The preprocessing is performed using the TensorFlow Datasets library.

# Model
The CNN model consists of several convolutional and pooling layers followed by two fully connected layers. The model is trained on the preprocessed dataset using the Adam optimizer and the sparse categorical crossentropy loss function. Dropout and data augmentation are used to prevent overfitting.

# Usage
Clone this repository to your local machine.
Install the required dependencies using pip install -r requirements.txt.
Run python train.py to train the model. The trained model will be saved in the "models" directory.
Run python evaluate.py to evaluate the model on the test set.
To use the model for inference, run python predict.py <image_path>, where <image_path> is the path to the image file you want to classify.

# Dependencies
TensorFlow 2.0 or higher
TensorFlow Datasets
Results
The model achieves an accuracy of 95% on the test set, with a loss of 0.14.

# Credits
Dataset: Horea Muresan, Mihai Oltean, and Fruit recognition from images using deep learning, Acta Univ. Sapientiae, Informatica Vol. 10, Issue 1, pp. 26-42, 2018.
Model architecture: based on the VGG16 architecture by Karen Simonyan and Andrew Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," arXiv:1409.1556
