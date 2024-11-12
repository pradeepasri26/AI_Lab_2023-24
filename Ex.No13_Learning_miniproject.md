# Ex.No: 10 Learning – Use Supervised Learning  
### DATE: 11/11/2024                                                                           
### REGISTER NUMBER : 212221220038
### AIM: 
To predict customer churn for a telecom company by analyzing customer behavior and usage patterns, enabling proactive retention strategies.
###  Algorithm:
1.Import Libraries: Load necessary libraries (Pandas, NumPy, scikit-learn, etc.) and ignore warnings.
2.Load Dataset: Load the CSV file, drop unnecessary columns, and convert data types as needed.
3.Handle Missing Values: Check for and fill missing values in TotalCharges.
4.Data Encoding: Encode categorical variables with LabelEncoder.
5.Data Splitting: Split data into training and testing sets.
6.Train Random Forest Model: Initialize and train a RandomForestClassifier on the training data.
7.Evaluate Random Forest Model: Predict and evaluate the model, showing accuracy, classification report, and confusion matrix.
8.Train Voting Classifier Ensemble: Combine GradientBoostingClassifier, LogisticRegression, and AdaBoostClassifier into a VotingClassifier, and train it.
9.Evaluate Voting Classifier: Predict and assess ensemble accuracy and confusion matrix.
10.End Program.
### Program:
```
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report

df = pd.read_csv('/content/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
df.shape
df = df.drop(['customerID'], axis = 1)
df.head()
df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
df.isnull().sum()
df.fillna(df["TotalCharges"].mean())numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_cols].describe()
df["Churn"][df["Churn"]=="No"].groupby(by=df["gender"]).count()
df["Churn"][df["Churn"]=="Yes"].groupby(by=df["gender"]).count()
model_rf = RandomForestClassifier(n_estimators=500 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
model_rf.fit(X_train, y_train)

# Make predictions
prediction_test = model_rf.predict(X_test)
print (metrics.accuracy_score(y_test, prediction_test))
print(classification_report(y_test, prediction_test))
plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, prediction_test),
                annot=True,fmt = "d",linecolor="k",linewidths=3)

plt.title(" RANDOM FOREST CONFUSION MATRIX",fontsize=14)
plt.show()	
from sklearn.ensemble import VotingClassifier
clf1 = GradientBoostingClassifier()
clf2 = LogisticRegression()
clf3 = AdaBoostClassifier()
eclf1 = VotingClassifier(estimators=[('gbc', clf1),
('lr', clf2), ('abc', clf3)], voting='soft')
eclf1.fit(X_train, y_train)
predictions = eclf1.predict(X_test)
print("Final Accuracy Score ")
print(accuracy_score(y_test, predictions))
plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, predictions),
                annot=True,fmt = "d",linecolor="k",linewidths=3)

plt.title("FINAL CONFUSION MATRIX",fontsize=14)
plt.show()
```
### Output:
![image](https://github.com/user-attachments/assets/e8a66d3c-9fc1-4054-8c30-1f0a716d9423)

![image](https://github.com/user-attachments/assets/02e6a5ac-c44a-4eff-9984-ec8d18e695e8)
![image](https://github.com/user-attachments/assets/b969d99d-a56b-401d-b216-a9b6e8b5e08f)

### Result:
Thus the system was trained successfully and the prediction was carried out.
