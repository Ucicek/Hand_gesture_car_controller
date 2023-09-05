# Hand-Gesture-Based Remote Car Controller 🚗🤚

## Overview

This project is an integration of machine learning and hardware to control a car using hand gestures. It provides a hands-on approach to solving real-world problems using Convolutional Neural Networks (CNN) and transfer learning models like VGG16 and RESNET50. The car itself is a DIY project built from scratch and is controlled through an Arduino.

## Table of Contents

1. [Model Training and Architecture](#model-training-and-architecture)
2. [Technology Stack](#technology-stack)
3. [Challenges](#challenges)
4. [Demo](#demo)

## Model Training and Architecture

### Dataset Preparation

1. **Data Collection**: Hand gesture images were collected using OpenCV. Each class of gesture was stored in separate folders to create a labeled dataset.
2. **Data Preprocessing**: Utilized PyTorch's data loaders to apply on-the-fly data augmentation, such as rotation and flipping, during training.

### Architectures Explored

1. **Basic CNN**: A simple 4-layer Convolutional Neural Network was designed as a baseline model. 
2. **Transfer Learning Models**: To improve performance, pretrained models such as VGG16 and Resnet50 were also used.

### Training Procedure

1. **Environment Setup**: Initially attempted to train on a local machine but moved to Google Colab and eventually AWS EC2 instances for better computational resources.
2. **Optimizer and Loss Function**: Used Adam optimizer with a learning rate of 0.001 and Cross-Entropy loss for classification.
3. **Early Stopping and Checkpointing**: To avoid overfitting, early stopping was implemented, and model checkpoints were saved at each epoch.
4. **Learning Rate Scheduling**: Implemented learning rate decay to refine the training.

### Challenges Faced During Training

1. **Overfitting**: Despite using data augmentation, the model was initially overfitting. Solved by introducing Dropout layers.
2. **Limited Data**: Had to be creative in generating more data and used MediaPipe for pre-processing.
3. **Convergence**: Faced issues with model convergence which led to further hyperparameter tuning.

### Evaluation Metrics

- Tracked the Loss and Accuracy for both training and validation sets.
- Utilized TensorBoard to visualize performance metrics in real-time.

### Fine-tuning and Iterative Improvement

- Conducted multiple rounds of fine-tuning to fix issues such as overfitting and non-convergence.
- Utilized the best-performing model for the final deployment on the car.




## Challenges

1. **Computation Power**: Limited by the computational power of my local machine, I utilized Google Colab and AWS EC2 instances for model training.
  
2. **Data Scarcity**: Solved the issue of limited data by applying MediaPipe for pre-processing and augmenting the dataset.

3. **Overfitting**: Resolved overfitting issues by adding Dropout layers, and increasing the amount of data.

## Technology Stack

- Python
- PyTorch
- OpenCV
- Arduino
- AWS
- Google Colab


## DEMO 

https://user-images.githubusercontent.com/77251886/197408183-1fb9be37-d97f-4cc8-bd69-aa5cc57a6fe9.mp4

