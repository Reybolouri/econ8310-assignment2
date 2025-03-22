#Assignment2
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb


train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
train_data = pd.read_csv(train_url)
test_data = pd.read_csv(test_url)

train_missing_values = train_data.isnull().sum()
test_missing_values = test_data.isnull().sum()

print("Train missing values:\n", train_missing_values)
print("Test missing values:\n", test_missing_values)


#Drop non predictive columns
train_data = train_data.drop(['id', 'DateTime'], axis=1)

#Separate target variable meal from predictors
X_train = train_data.drop('meal', axis=1)
y_train = train_data['meal']
#split the data
train_X, val_X, train_Y, val_Y = train_test_split(X_train, y_train, test_size=0.33, random_state=42)


xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Fit the model on training set
modelFit = xgb_model.fit(train_X, train_Y)

# Evaluate model on training set 
train_preds = modelFit.predict(train_X)
train_accuracy = accuracy_score(train_Y, train_preds)
print("Training Accuracy: {:.2f}%".format(100 * train_accuracy))

# Evaluate model on validation set
val_preds = modelFit.predict(val_X)
val_accuracy = accuracy_score(val_Y, val_preds)
print("Validation Accuracy: {:.2f}%".format(100 * val_accuracy))


# Fit the Final Model on the Full Training Data
final_model = xgb_model.fit(X_train, y_train)

# -------process test data and make predictions
test_data = test_data.drop(['id', 'DateTime'], axis=1)
if 'meal' in test_data.columns:
    test_data = test_data.drop('meal', axis=1)

X_test = test_data[X_train.columns]


predictions = final_model.predict(X_test)
pred = predictions.astype(int).tolist()

print("Number of predictions:", len(pred))

print("First 10 predictions:", pred[:10])