🕵️‍♂️ Credit Card Fraud Detection
A machine learning project to detect fraudulent credit card transactions using classification algorithms and techniques to handle imbalanced data.

📚 Overview
Credit card fraud is a critical issue in financial systems. This project uses machine learning to classify transactions as legitimate or fraudulent, with a focus on handling dataset imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

🚀 Objectives
Build a classification model to identify fraudulent transactions.

Explore and compare algorithms like Logistic Regression and Decision Trees.

Handle class imbalance using SMOTE.

Evaluate model performance using precision, recall, F1-score, and ROC-AUC.

🛠️ Technologies Used
Python

Scikit-learn

imbalanced-learn

Pandas & NumPy

Matplotlib & Seaborn (for visualization)

📊 Workflow
Data Loading: Load the dataset of credit card transactions.

Preprocessing: Scale features, handle missing values.

Handling Imbalance: Apply SMOTE to balance the class distribution.

Model Training:

Train Logistic Regression and Decision Tree models.

Tune hyperparameters to optimize performance.

Evaluation:

Use predict() for class labels and predict_proba() for probabilities.

Generate confusion matrix, ROC curve, and classification report.

📈 Evaluation Metrics
Accuracy

Precision & Recall

F1 Score

ROC-AUC

🙌 Credits
Scikit-learn

Imbalanced-learn

Original dataset on Kaggle

Dataset Download : 
import kagglehub

# Download latest version
path = kagglehub.dataset_download("kartik2112/fraud-detection")

print("Path to dataset files:", path)
