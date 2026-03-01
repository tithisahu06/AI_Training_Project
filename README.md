Deployed Link: https://ccfdai.streamlit.app/

Credit Card Fraud Detection using Machine Learning

Project Overview
This project focuses on detecting fraudulent credit card transactions using Machine Learning algorithms. The objective is to build a classification model that can accurately identify whether a transaction is fraudulent or genuine, helping financial institutions minimize losses due to fraud.

Due to the highly imbalanced nature of transaction data, special preprocessing and evaluation techniques were applied to improve model performance.

Objectives:
Analyze and understand transaction dataset
Handle imbalanced data effectively
Train multiple ML classification models
Compare model performance
Identify the best-performing model

Dataset Information:
The dataset used for this project is the Credit Card Fraud Detection Dataset available on Kaggle.
It contains transactions made by European cardholders.
Features V1, V2, V3 ... V28 are the result of PCA transformation (for confidentiality reasons).
Time and Amount are original features.

Class is the target variable:
0 → Genuine Transaction
1 → Fraudulent Transaction

The dataset is highly imbalanced (fraud cases are very rare compared to normal transactions).

Technologies Used:
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook

Project Workflow
1) Data Preprocessing:
Checked for missing values
Scaled the Amount feature
Handled class imbalance
Split dataset into training and testing sets

2️)Exploratory Data Analysis (EDA):
Count plot of fraud vs genuine transactions
Correlation heatmap
Distribution analysis

3)Model Building:
The following models were implemented:
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
K-Nearest Neighbors

4)Model Evaluation:
Models were evaluated using-
Accuracy Score
Precision
Recall
F1-Score
Confusion Matrix

Since the dataset is imbalanced, Recall and F1-score were considered more important than accuracy.

Results:
After comparison, the best-performing model was selected based on Recall and F1-score to ensure maximum fraud detection with minimal false negatives.

Key Learnings:
Handling imbalanced datasets
Importance of evaluation metrics in classification problems
Model comparison and performance analysis
Real-world application of machine learning in finance



Author
Tithi Sahu
If you like this project, feel free to star the repository!

