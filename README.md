# Rv-systems
#Machine Learning Classification of Vibration Data
This repository contains code to classify vibration data from various machines using different machine learning models. The code includes data loading, preprocessing, model training, hyperparameter tuning, and evaluation.

# Project Overview
The objective of this project is to classify time-series vibration data from 12 different machines using several classification models. The data is processed, features are extracted, and various models are trained to achieve classification accuracy. The models evaluated include Support Vector Machine (SVM), k-Nearest Neighbors (k-NN), Naive Bayes, and Decision Tree. An ensemble voting classifier is also used to combine the predictions from these models.

# Directory Structure
dataset_dir/: Directory containing the .dat files with the vibration data.

# Requirements
To run this code, ensure you have the following Python packages installed:

numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm

# Pipeline Overview
This repository contains a comprehensive machine learning pipeline for classifying time-series vibration data. The pipeline includes data loading, preprocessing, feature extraction, model training, and evaluation.

# Data Loading
The script loads time-series vibration data from .dat files located in the RV_Systems_ML_Training_Sets directory. Each file is accessed using numpy.fromfile, and data is mapped based on its index.

# Data Preprocessing
The preprocessing steps include:

# Downsampling: To expedite processing, the data is randomly downsampled.
Handling Missing Values: Missing values are replaced using numpy.nan_to_num.
Feature Extraction: The real and imaginary components of the vibration data are separated. Statistical features, such as mean and standard deviation, are computed and appended to the data.

# Feature Extraction
Additional statistical features (mean and standard deviation) are computed and appended to the vibration data. The resulting feature set is then standardized using StandardScaler.

# Model Training and Evaluation
The pipeline includes the training and evaluation of the following machine learning models:

# k-Nearest Neighbors (k-NN): Tuned using GridSearchCV.
# Naive Bayes (NB): Uses a default Gaussian Naive Bayes classifier.
# Decision Tree (DT): A decision tree with restricted depth for complexity management.

Each model's performance is assessed using accuracy, confusion matrix, classification report. You can install these packages using pip:

pip install -r requirements.txt

# How to Use
Download and Prepare Data: Ensure the .dat files are placed in the RV_Systems_ML_Training_Sets directory.
Install Dependencies: Install the required Python packages.

## Notes
The pipeline effectively classifies vibration data, achieving high accuracy with the ensemble voting classifier.
Further tuning and exploration of additional features can potentially improve performance.
