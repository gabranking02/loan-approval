
#import the libraries
import pandas as pd
import numpy as np
dataset = pd.read_csv('train_ctrUa4K.csv')

# drop the null values
dataset.dropna(how='any', inplace=True)

#encoding the dependent variabl:
from sklearn.preprocessing import LabelEncoder
le  = LabelEncoder()
y = le.fit_transform(dataset.Loan_Status)

from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size=0.2,random_state=0)

train_x = train.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
train_y = train['Loan_Status']

test_x = test.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
test_y = test['Loan_Status']

# encode the data
train_x = pd.get_dummies(train_x)
test_x = pd.get_dummies(test_x)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0, solver='lbfgs')
classifier.fit(train_x, train_y)

y_pred = classifier.predict(test_x)

from sklearn import metrics
print('accruracy_score:',metrics.accuracy_score(test_y, y_pred))

accruracy_score: 0.7291666666666666
