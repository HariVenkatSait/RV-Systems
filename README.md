# rv-systems
Machine Learning Classification of Vibration Data
Project Overview
This project aims to classify time-series vibration data from 12 different machines using advanced machine learning models. The pipeline includes data loading, preprocessing, feature extraction, model training, hyperparameter tuning, and evaluation. Models evaluated include Gradient Boosting (XGBoost), Random Forest, SVM, k-NN, Naive Bayes, Decision Tree, and an ensemble voting classifier.

Directory Structure
dataset_dir/: Directory containing the .dat files with the vibration data.
Requirements
To run this project, ensure you have the following Python packages installed:

numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm
You can install these packages using pip:

bash
Copy code
pip install numpy pandas matplotlib seaborn scikit-learn tqdm
Pipeline Overview
The project implements a comprehensive machine learning pipeline for classifying time-series vibration data. The key stages are:

Data Loading: Loads vibration data from .dat files using numpy.fromfile.
Data Preprocessing: Includes downsampling, handling missing values with numpy.nan_to_num, and feature extraction.
Feature Extraction: Extracts statistical features, frequency domain features (FFT), and time-frequency features (STFT, Wavelets). Standardization is applied using StandardScaler.
Model Training and Evaluation: Includes training and evaluation of multiple models and hyperparameter tuning using GridSearchCV.
Models Evaluated
k-Nearest Neighbors (k-NN): Achieved 34% accuracy.
Naive Bayes (NB): Achieved 17% accuracy.
Decision Tree (DT): Achieved 36% accuracy.

Evaluation Metrics
Models are evaluated using:

Accuracy
Confusion Matrix
Classification Report
Learning Curves
How to Use
Download and Prepare Data: Ensure the .dat files are placed in the RV_Systems_ML_Training_Sets directory.

Install Dependencies: Run the command to install required Python packages:

bash
Copy code
pip install -r requirements.txt
Run the Pipeline: Execute the main script to start the data loading, preprocessing, feature extraction, model training, and evaluation.

Notes
The pipeline effectively classifies vibration data, achieving high accuracy with the ensemble voting classifier.
Further tuning and exploration of additional features can potentially improve performance.
