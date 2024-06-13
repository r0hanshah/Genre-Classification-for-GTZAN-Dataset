# Genre Classification for GTZAN Dataset

## Project Overview

This project aims to classify a song’s genre based on given audio samples using a Convolutional Recurrent Neural Network (CRNN). The primary dataset utilized is the GTZAN Dataset, which comprises audio files from various music genres. This is a supervised multi-class classification task.

## Team Members

- **Rohan Shah (POC)** - [rohanshah1@ufl.edu](mailto:rohanshah1@ufl.edu)
- Nishant Nagururu - [nishant.nagururu@ufl.edu](mailto:nishant.nagururu@ufl.edu)
- Daniel Parra - [dparra1@ufl.edu](mailto:dparra1@ufl.edu)
- Alvin Wong - [alvinwong@ufl.edu](mailto:alvinwong@ufl.edu)

## Table of Contents

1. [Introduction](#introduction)
2. [Approach: Dataset & Pipeline](#approach-dataset--pipeline)
3. [Evaluation Methodology](#evaluation-methodology)
4. [Results](#results)
5. [Conclusions](#conclusions)
6. [References](#references)

## Introduction

This project aims to accurately classify music genres using audio samples. The audio files, which come from multiple genres, are transformed into spectrograms and segmented before being processed by a CRNN model. 

## Approach: Dataset & Pipeline

### Dataset

- **Introduction**: We used the GTZAN Dataset, which includes 10 genres with 100 audio files each. Each audio file is represented by a visual spectrogram image.
- **Size**: 6000 instances, 10 music genres, 1.41 GB in total.
- **Data Source**: [GTZAN Dataset on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data)

### Pipeline

- **Model Architecture**: A CRNN model was designed with multiple CNN and pooling layers followed by a GRU-based RNN layer.
- **Data Processing**: Spectrogram images were cropped and grayscaled. Each spectrogram was split into 6 sub-images, representing 5 seconds of audio each.
- **Training**: The dataset was split into training, validation, and test sets (80/10/10). Data augmentation techniques were applied to prevent overfitting.
- **Evaluation**: Categorical cross-entropy loss was used, with softmax activation for output and ReLU for convolutional layers.

## Evaluation Methodology

- **Accuracy**: Overall model accuracy was computed as the average per-class accuracy.
- **F1 Score**: Precision and recall metrics were calculated to determine the F1 score.
- **Baselines**: 
  - **Random Guessing**: 10% accuracy.
  - **CNN Baseline**: 85% accuracy.
  - **RNN Baseline**: 75% accuracy.

## Results

- **Training and Validation Accuracy**: Our model achieved a training accuracy of approximately 78.13%.
- **Confusion Matrix**: The model performed well on genres with clear patterns like Reggae and Classical, but struggled with similar-sounding genres like Rock and Country.
- **Performance**: Although our CRNN model did not surpass the 85% accuracy baseline, it significantly outperformed random guessing.

## Conclusions

Our project demonstrates that CNNs are effective in classifying music genres from spectrograms. Future improvements could include expanding the dataset, experimenting with different machine learning methods, and refining the CRNN architecture. Additional strategies such as ensemble methods and multiple spectrogram types could further enhance performance.

## References

1. [GTZAN Dataset on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data)
2. [A Globally Regularized Joint Neural Architecture for Music Classification](https://www.researchgate.net/publication/347771583_A_Globally_Regularized_Joint_Neural_Architecture_for_Music_Classification)
3. [AltexSoft Blog on Audio Analysis](https://www.altexsoft.com/blog/audio-analysis/)
