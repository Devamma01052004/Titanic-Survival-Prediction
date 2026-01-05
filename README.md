# Titanic-Survival-Prediction
Machine Learning project using Titanic dataset
# Titanic Survival Prediction 🚢

Predict whether a passenger survived the Titanic disaster using Machine Learning.

## Project Overview

This project uses the Titanic dataset from Kaggle to predict passenger survival. Features include:

- Passenger Class (`Pclass`)
- Name, Sex, Age
- Number of Siblings/Spouses (`SibSp`) and Parents/Children (`Parch`)
- Ticket number, Fare, Embarked location

We build a Machine Learning model to predict survival based on these features.

---

## Project Structure

titanic-ml-project/
│
├── data/
│ ├── train.csv # Original training dataset
│ ├── test.csv # Original test dataset
│ ├── train_cleaned.csv # Cleaned training data
│ └── test_cleaned.csv # Cleaned test data
│
├── notebooks/
│ └── titanic_model.ipynb # Jupyter notebook for analysis & model building
│
├── models/
│ └── titanic_model.pkl # Saved trained ML model
│
├── app.py # Streamlit app to predict survival
├── gender_submission.csv # Sample Kaggle submission
├── titanic_submission.csv # Your final predictions
└── README.md # Project documentation

---

## How to Run the Project

### 1. Using Jupyter Notebook
1. Open `titanic_model.ipynb`.
2. Explore, clean, and preprocess the data.
3. Train and evaluate the ML model.
4. Save the trained model as `titanic_model.pkl`.

### 2. Using Streamlit App
1. Install required libraries:

```bash
pip install streamlit pandas scikit-learn joblib

streamlit run app.py
