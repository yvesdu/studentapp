import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score

data = pd.read_csv("~/Desktop/studentapp/xAPI-Edu-Data.csv")

data.head(10)

for column in ['Class']:
    data[column] = data[column].astype('category')
    data[column] = data[column].cat.codes
    
data.head()

data.columns

data = pd.get_dummies(data)

data.head()

data.columns

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.drop('Class',axis=1),data['Class'], test_size=0.33, random_state=1)

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)

labels = model.predict(X_test)
labels_1 = model.predict_proba(X_test)

from sklearn.metrics import log_loss
log_loss(y_test, labels_1)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, labels)

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,)
plt.xlabel('true label')
plt.ylabel('predicted label');

from sklearn.linear_model import LogisticRegression
model2 = LogisticRegression(C=100.0,random_state=0)
model2.fit(X_train,y_train)

labels2 = model2.predict(X_test)
labels_2 = model2.predict_proba(X_test)

log_loss(y_test, labels_2)

mat2 = confusion_matrix(y_test, labels2)
sns.heatmap(mat2.T, square=True, annot=True, fmt='d', cbar=False,)
plt.xlabel('true label')
plt.ylabel('predicted label');

accuracy_score(y_test, labels2)

from sklearn.svm import SVC
model3 = SVC(kernel='linear', C=1.0, random_state=0, probability=True)
model3.fit(X_train,y_train)

labels3 = model3.predict(X_test)
labels_3 = model3.predict_proba(X_test)

mat3 = confusion_matrix(y_test, labels3)
sns.heatmap(mat3.T, square=True, annot=True, fmt='d', cbar=False,)
plt.xlabel('true label')
plt.ylabel('predicted label');

accuracy_score(y_test, labels3)
log_loss(y_test, labels_3)

model4 = SVC(kernel='rbf', C=1.0, random_state=0, gamma=100.0, probability=True)

model4.fit(X_train,y_train)

labels4 = model4.predict(X_test)
labels_4 = model4.predict_proba(X_test)

mat4 = confusion_matrix(y_test, labels4)
sns.heatmap(mat4.T, square=True, annot=True, fmt='d', cbar=False,)
plt.xlabel('true label')
plt.ylabel('predicted label');

log_loss(y_test,labels_4)

from sklearn.tree import DecisionTreeClassifier
model5 = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)

model5.fit(X_train,y_train)
labels5 = model5.predict(X_test)
labels_5 = model5.predict_proba(X_test)
mat5 = confusion_matrix(y_test, labels5)
sns.heatmap(mat5.T, square=True, annot=True, fmt='d', cbar=False,)
plt.xlabel('true label')
plt.ylabel('predicted label');

log_loss(y_test,labels_5)

from sklearn.ensemble import RandomForestClassifier

model6 = RandomForestClassifier(criterion='entropy',
                               n_estimators=29,
                               random_state=0)
model6.fit(X_train,y_train)

labels6 = model6.predict(X_test)

labels_6 = model6.predict_proba(X_test)

mat6 = confusion_matrix(y_test, labels6)
sns.heatmap(mat6.T, square=True, annot=True, fmt='d', cbar=False,)
plt.xlabel('true label')
plt.ylabel('predicted label');

log_loss(y_test,labels_6)

from sklearn.neighbors import KNeighborsClassifier
model7 = KNeighborsClassifier(n_neighbors=5,p=2,
                             metric='minkowski')
                             
model7.fit(X_train,y_train)

labels7 = model7.predict(X_test)

labels_7 = model7.predict_proba(X_test)

mat7 = confusion_matrix(y_test, labels6)
sns.heatmap(mat7.T, square=True, annot=True, fmt='d', cbar=False,)
plt.xlabel('true label')
plt.ylabel('predicted label');

log_loss(y_test,labels_7)

