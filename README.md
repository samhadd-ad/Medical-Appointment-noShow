# ITEC 3040 Final Project - Group G

This project predicts whether a patient will miss their medical appointment using machine learning. It includes dataset preprocessing, feature engineering, model training, evaluation, and an interactive Streamlit app for real-time predictions.(KaggleV2-May-2016.csv)

## Group Members

- Jiaqi Fu (218028415)
- Dominic Gopalakrishnan (220124392)
- Suning Gu (218340364)
- Shushanth Gudimalla (218943225)
- Sam Haddad (219930445)
# Features

Data cleaning and preprocessing

Exploratory visualizations

Machine learning model trained to predict no-shows

Saved model artifacts (.pkl files)

Streamlit web interface for user input and predictions

Reproducible environment with requirements.txt
## 1. Project Structure

```text
.
├── KaggleV2-May-2016.csv     # Dataset
├── analysis.py               # Data cleaning + EDA + modelling
├── app.py                    # Streamlit UI application
├── requirements.txt          # Python dependencies
└── README.md                 # This file


# How It Works

The raw dataset is cleaned and transformed.

Features are encoded and scaled.

A classification model is trained to detect patterns related to no-shows.

The trained model and feature structure are saved.

The Streamlit app loads these artifacts and predicts based on user inputs.

# Technologies Used

Python

Pandas, NumPy

Scikit-learn

Streamlit

Matplotlib / Seaborn
