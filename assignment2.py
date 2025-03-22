#Assignment2
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"



train_data = pd.read_csv(train_url)
test_data = pd.read_csv(test_url)


train_missing_values = train_data.isnull().sum()
print("Train set missing values:\n", train_missing_values)
test_missing_values = test_data.isnull().sum()
print("Train set missing values:\n", test_missing_values)

target = 'meal'
features = train_data.columns.drop(target)
X = train_data[features]
y = train_data[target]

X = pd.get_dummies(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42, n_estimators=100)

modelFit_split = model.fit(X_train, y_train)

# in sample/training accuracy
train_preds = modelFit_split.predict(X_train)
in_sample_accuracy = accuracy_score(y_train, train_preds)
print("In sample (training) accuracy:", in_sample_accuracy)

#out of sample/validation accuracy
val_preds = modelFit_split.predict(X_val)
out_of_sample_accuracy = accuracy_score(y_val, val_preds)
print("Out of sample (validation) accuracy:", out_of_sample_accuracy)

modelFit = model.fit(X, y)


#---------------------

# Prep the test data.
if 'meal' in test_data.columns:
    X_test = test_data.drop(columns=['meal'])
else:
    X_test = test_data.copy()

#Apply one hot encoding to test data
X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=X.columns, fill_value=0)


pred = modelFit.predict(X_test)

print("Number of predictions:", len(pred))
print(pd.DataFrame({'predicted_meal': pred}).head())