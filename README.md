# Breast_Cancer_Analysis

This project leverages machine learning techniques, specifically Artificial Neural Networks (ANN), for predicting breast cancer malignancy using the popular Breast Cancer Wisconsin dataset. It also includes a user-friendly Streamlit web application for interaction and prediction.

## Setup

- Clone the repository. (git clone <repository_link>)
- Create a virtual environment and install required packages.(On Windows use venv\Scripts\activate)
- python -m venv venv
- pip install -r requirements.txt
- Run the Streamlit app locally: ( streamlit run streamlit.py

## Features

- **Data Preprocessing and Feature Selection:** The project includes loading the breast cancer dataset, handling missing values, and selecting the most relevant features using `SelectKBest`.
- **Optimized ANN Model*
  - Hyperparameter Tuning: Grid Search Cross-Validation optimizes hidden layers, activation functions, 
    solvers, and regularization.
  - Training: Implements MLPClassifier from scikit-learn for accurate prediction.
- **Interactive Predictions using Streamlit App*
  - The Streamlit app provides a user-friendly interface for predictions about whether a tumor is malignant 
    or benign. based on user inputs.

## Project Structure

- ├── data_preparation.py      
  - # Loads and preprocesses dataset
- ├── feature_selection.py      
  - # Performs feature selection
- ├── model_selection.py        
  - # Hyperparameter tuning with Grid Search
  - # ANN training and evaluation
- ├── streamlit.py
  - # Interactive Streamlit app
- ├── breast_cancer_data_.csv  # 
  - # Preprocessed dataset
- ├── requirements.txt          
  - # Project dependencies
- └── README.md                 
  - # Project documentation

## Requirements

- Python 3.x
- Streamlit
- scikit-learn
- pandas
- joblib

![Output](https://github.com/user-attachments/assets/33756f6b-7d43-422b-be18-e541101ab9eb)
![Output1](https://github.com/user-attachments/assets/1de0c3ef-b8c2-4621-8ae5-13a6b74c2c30)



