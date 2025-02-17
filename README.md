# DS-and-ML-mini-projects
## Titanic Survival Prediction

### Project Overview

This project predicts passenger survival on the Titanic using machine learning models. The dataset contains various features such as age, gender, class, and ticket fare, which help determine survival chances.

Dataset

Source: Kaggle Titanic Dataset

Features: Passenger ID, Name, Age, Gender, Pclass, Fare, Embarked, etc.

Target Variable: Survived (1 = Survived, 0 = Not Survived)

Approach

Data Preprocessing

Handle missing values (Age, Cabin, Embarked)

Convert categorical variables to numerical

Feature scaling and encoding

Feature Engineering

Create new features (Family Size, Title Extraction)

Drop irrelevant columns

Model Training

Train multiple models: Logistic Regression, Random Forest, XGBoost

Hyperparameter tuning with GridSearchCV

Model evaluation using accuracy, precision, recall, and F1-score

Results

Best Model: Random Forest (Accuracy: 82%)

Key insights: Women and children had higher survival rates

Technologies Used

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Solar Irradiance Prediction

Project Overview

This project forecasts solar irradiance using regression models based on weather data.

Dataset

Features: Temperature, Humidity, Wind Speed, Cloud Cover

Target Variable: Solar Irradiance (W/m²)

Approach

Data Cleaning & Preprocessing

Handle missing values

Feature selection

Normalization

Model Selection

Linear Regression, Decision Tree, Random Forest, XGBoost

Hyperparameter tuning

Evaluation Metrics

Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² Score

Results

Best Model: XGBoost (R² Score: 0.92)

Technologies Used

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Digit Recognizer

Project Overview

Developed a deep learning model to classify handwritten digits using the MNIST dataset.

Dataset

Source: MNIST Handwritten Digits Dataset

Features: 28x28 pixel grayscale images

Target Variable: Digits (0-9)

Approach

Data Preprocessing

Normalize pixel values

Data augmentation using ImageDataGenerator

Model Development

CNN using TensorFlow/Keras

Layers: Convolutional, Pooling, Fully Connected

Evaluation

Accuracy, Confusion Matrix, Loss Graphs

Results

Model Accuracy: 98%

Technologies Used

Python, TensorFlow/Keras, NumPy, Matplotlib

