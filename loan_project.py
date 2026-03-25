import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#1) Load Data
# Loading the dataset
data = pd.read_csv(r'C:\Users\nitin\Desktop\0\Project\loan_data.csv')

# Checking for nulls first thing
print("Missing values in each column:")
print(data.isnull().sum())

#2) Quick EDA (Exploratory Data Analysis)
# I want to see if Credit History is the main factor for approval
plt.figure(figsize=(8,4))
sns.countplot(x='Credit_History', hue='Loan_Status', data=data, palette='magma')
plt.title('Loan Status based on Credit History')
plt.show()

#3)Cleaning & Preprocessing
# Dealing with missing values manually
# For numbers, I'll use the median. For text, I'll use the mode.
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])

# Categorical nulls
cols_to_fix = ['Gender', 'Married', 'Dependents', 'Self_Employed']
for col in cols_to_fix:
    data[col] = data[col].fillna(data[col].mode()[0])

#4)Feature Engineering
# Combining Income because usually, the bank looks at the total household income
data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
# Now we can drop the individual ones and the ID
data.drop(['Loan_ID', 'ApplicantIncome', 'CoapplicantIncome'], axis=1, inplace=True)

# Converting text categories to numbers
le = LabelEncoder()
cat_features = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status', 'Dependents']
for i in cat_features:
    data[i] = le.fit_transform(data[i].astype(str))

#5)Splitting Data
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#6) Model Testing (Trial and Error)
# First, I'll try a simple Logistic Regression as a baseline
lr_model = LogisticRegression(max_iter=10000)
lr_model.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
print(f"\nLogistic Regression Accuracy: {lr_acc:.4f}")

# Now, trying Random Forest to see if it performs better
# I'll limit the depth to 5 so it stays simple and doesn't overfit
rf_model = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"Random Forest Accuracy: {rf_acc:.4f}")

#7) Final Evaluation
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, rf_preds))

# Let's visualize the confusion matrix
sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 8) Testing with External Data
def test_new_application(gender, married, dep, edu, self_emp, loan_amt, term, credit_hist, area, app_inc, co_inc):
    total_inc = app_inc + co_inc
    sample_data = pd.DataFrame({
        'Gender': [1 if gender == 'Male' else 0],
        'Married': [1 if married == 'Yes' else 0],
        'Dependents': [dep], # Enter as '0', '1', '2', or '3'
        'Education': [0 if edu == 'Graduate' else 1],
        'Self_Employed': [1 if self_emp == 'Yes' else 0],
        'LoanAmount': [loan_amt],
        'Loan_Amount_Term': [term],
        'Credit_History': [credit_hist],
        'Property_Area': [2 if area == 'Urban' else (1 if area == 'Semiurban' else 0)],
        'Total_Income': [total_inc]
    })  
    # Ensure column order matches training data
    sample_data = sample_data[X.columns]
    # Predict using trained Random Forest model
    res = rf_model.predict(sample_data)
    return "Approved" if res[0] == 1 else "Rejected"

print("\n    Manual Test Case    ")
print("Result:", test_new_application('Male', 'Yes', '0', 'Graduate', 'No', 120, 360, 1.0, 'Urban', 5000, 2000))
