# Neural Network Biometric Identification & Authentication

## Overview

This repository contains a machine learning project developed for the USP Machine Learning course under Professor Clodoaldo. The objective of the project is to implement a neural network from scratch, without relying on high-level machine learning libraries, to perform biometric identification and authentication using facial image data.

The project utilizes the CelebA dataset to train and evaluate models on two related tasks:
1. Authentication: verifying whether a given input image belongs to a claimed identity.
2. Identification: determining the identity of the person in a given image from a set of known identities.

All components — including data preprocessing, model architecture, forward and backward propagation, loss computation, and evaluation — were manually implemented to reinforce understanding of core machine learning concepts.

## Dataset

The project uses the CelebA dataset, a large-scale face attributes dataset containing over 200,000 images of celebrities, each labeled with identity information. This dataset provides sufficient diversity and volume for training neural networks focused on biometric tasks.

The dataset must be downloaded separately and organized according to the directory structure described in the project documentation.



## Implementation Details

The neural network was implemented entirely from scratch, including:
- Network architecture definition
- Forward propagation
- Backpropagation and gradient computation
- Loss and evaluation metrics
- Training and testing loops

No high-level machine learning libraries such as TensorFlow, PyTorch, or scikit-learn were used.

## Tasks

- Biometric Authentication: verifies if an image matches a claimed identity.
- Biometric Identification: predicts the identity associated with a given facial image.

## Report

A comprehensive PDF report is included in the repository, detailing:
- Problem formulation
- Dataset analysis
- Model design
- Training methodology
- Experimental results
- Conclusions

## Technologies

- Python
- NumPy
- Custom neural network implementation
- CelebA dataset
