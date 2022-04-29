import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load the file
df = pd.read_csv('breast-cancer-wisconsin.csv')
# replace ?
df.replace("?",0, inplace=True) 

# Name the columns
names = ['code', 'clump-thickness', 'cell-size', 'cell-shape', 'marginal-adhesion','single-epithelial', 'bare-nuclei', 'bland-chromatin', 'normal-nucleoli', 'mitosis', 'Class']
df.columns = names

# Assign to X and y
X = df.iloc[:,1:10]
y = df.iloc[:,10]

#split train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)

# fit the model
clf = MLPClassifier(hidden_layer_sizes = [10,10,10], max_iter = 500).fit(X_train, y_train)

# cross validation score for 5 folds
c_val_score = cross_val_score(clf, X, y, cv=5);
for score in c_val_score:
    print(f'cross validation score = {score}')

# Predict the train and test
y_pred_test = clf.predict(X_test)
y_pred_train = clf.predict(X_train)

# Run confusion matrix
conf_train = confusion_matrix(y_train, y_pred_train)
conf_test = confusion_matrix(y_test, y_pred_test)

# Run classification report
cl_rep_train = classification_report(y_train, y_pred_train)
cl_rep_test = classification_report(y_test, y_pred_test)

print('Confusion Matrix train: \n')
print(conf_train)
print('\nConfusion Matrix test: \n')
print(conf_test)
print('\nClassification Report train: \n')
print(cl_rep_train)
print('\nClassification Report test: \n')
print(cl_rep_test)