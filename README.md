#Overview:

Financial distress and bankruptcy prediction are critical for investors, banks, and businesses to minimize risk.

This project builds a machine learning-based bankruptcy prediction system that analyzes financial data and predicts whether a company is likely to go bankrupt.

The model compares multiple algorithms:

Logistic Regression
Random Forest
XGBoost

to identify the most accurate and reliable approach.
Problem Statement

Traditional financial analysis is time-consuming and often subjective.
This project aims to:

Automate bankruptcy prediction
Provide data-driven insights
Improve decision-making for financial stakeholders

Key Features
Multiple ML model implementation and comparison
Feature importance analysis
Data preprocessing (handling missing values, normalization)
Model evaluation using metrics like accuracy, precision, recall, F1-score
Confusion matrix visualization
🛠️ Tech Stack
Language: Python
Libraries: Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn
Tools: Jupyter Notebook / VS Code

Dataset
Dataset contains financial indicators such as:
Profitability ratios
Liquidity ratios
Debt ratios
(Mention dataset source here – Kaggle/UCI/etc.)

Installation & Setup
git clone https://github.com/your-username/bankruptcy-prediction.git
cd bankruptcy-prediction
pip install -r requirements.txt

Run the notebook or script:

python main.py

Model Implementation
1. Logistic Regression
Baseline model for binary classification
Easy to interpret
2. Random Forest
Handles non-linearity well
Reduces overfitting via ensemble learning
3. XGBoost
Gradient boosting technique
High performance and accuracy
