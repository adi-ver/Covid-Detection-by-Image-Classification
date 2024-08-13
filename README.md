# X-ray Image Classification Model for COVID-19 Detection
This project utilizes Convolutional Neural Networks (CNN) to classify chest X-ray images as either COVID-19 positive or normal. The model is trained on a dataset of chest X-ray images, distinguishing between COVID-infected cases and those that are not.

__Disclaimer: This model is not intended for real-life diagnostic purposes and should only be used for educational purposes.__

### Introduction
The COVID-19 pandemic has highlighted the need for rapid and accurate diagnostic tools. This project leverages deep learning to develop a model that can assist in the detection of COVID-19 from chest X-ray images. The model is built using TensorFlow and Keras, with a focus on achieving high accuracy through effective data preprocessing and model tuning.

### Dataset
The dataset used for this project contains chest X-ray images, divided into two categories:
- COVID-19: Chest X-ray images of COVID-19 positive cases.
- Normal: Chest X-ray images of healthy cases.
- The dataset is structured into training and test directories, each containing subdirectories for COVID-19 and normal images.
- Data Source: https://github.com/education454/datasets.git

### Model Architecture
The model is a Convolutional Neural Network (CNN) built using TensorFlow and Keras. The architecture consists of:
- Convolutional Layers: Two convolutional layers with ReLU activation and max-pooling.
- Dropout Layers: Dropout layers to prevent overfitting.
- Dense Layer: A fully connected layer with 256 nodes and ReLU activation.
- Output Layer: A single node with sigmoid activation for binary classification.

### Data Preprocessing & Augmentation
To enhance the model's performance, the following data preprocessing and augmentation techniques were applied:
- Rescaling: All images were rescaled by a factor of 1./255.
- Augmentation: Training data was augmented with random zooms and horizontal flips.
- Batch Generation: Data was split into training, validation, and test sets, with batches generated using the ImageDataGenerator class.

### Training
The model was trained for 5 epochs using the Adam optimizer with a learning rate of 0.001. The loss function used was binary crossentropy, and the model was evaluated on both training and validation sets during training.

### Evaluation
The model's performance was evaluated using accuracy and loss metrics. The evaluation process involved:
- Plotting training and validation loss across epochs.
- Plotting training and validation accuracy across epochs.
- Testing the model on the test set to obtain final accuracy and loss.
### Usage
To use the model for predicting COVID-19 from new chest X-ray images:

- Upload the chest X-ray image.
- The model will output whether the image is classified as COVID-19 positive or normal.
### Results
The model achieved a significant level of accuracy in distinguishing between COVID-19 positive and normal chest X-ray images. The final test accuracy and loss are provided in the project output.
