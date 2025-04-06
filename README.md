# Heart-Stroke
Overview
This project implements and compares multiple machine learning models to predict the likelihood of stroke based on patient data. Using a comprehensive healthcare dataset with various demographic and health metrics, the models achieve up to 95.1% accuracy in identifying potential stroke cases.

Dataset
The dataset (healthcare-dataset-stroke-data.csv) contains the following features:
Gender
Age
Hypertension
Heart disease history
Marital status
Work type
Residence type
Average glucose level
BMI (Body Mass Index)
Smoking status
Stroke occurrence (target variable)

Methodology
Data Preprocessing
Removed ID column (non-predictive)
Handled missing BMI values by imputing with the mean
Encoded categorical variables
Performed normality testing on numerical features
Applied feature scaling using MinMaxScaler

Exploratory Data Analysis
Statistical summary of the dataset
Correlation analysis
Distribution visualization of key features
Relationship between stroke occurrence and demographic factors

Models Implemented
The project compares four different classification algorithms:
Logistic Regression
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Decision Tree Classifier
Each model was evaluated using a 10-fold cross-validation repeated 5 times to ensure robust performance metrics.

Results
Logistic Regression: 95.1% accuracy
SVM: 95.1% accuracy
KNN: 94.9% accuracy
Decision Tree Classifier: 91.1% accuracy
