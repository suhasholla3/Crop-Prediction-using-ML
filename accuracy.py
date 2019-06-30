import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

acc=list()

train=pd.read_csv('training.csv')
test=pd.read_csv('test.csv')

train.drop(['Crop'],axis=1,inplace=True)
test.drop(['Crop'],axis=1,inplace=True)

#using KNN
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(train.drop(['Value'],axis=1),train['Value'])
y_pred = classifier.predict(test.drop(['Value'],axis=1))
from sklearn.metrics import accuracy_score
print("Accuracy using KNN")
accuracy=accuracy_score(test['Value'],y_pred)
print(accuracy*100)
acc.append(round(accuracy,2))

#using LOGREG
from sklearn import linear_model
reg=linear_model.LogisticRegression()
reg.fit(train.drop(['Value'],axis=1),train['Value'])
y_pred=reg.predict(test.drop(['Value'],axis=1))
print("Accuracy using Logistic Regression")
accuracy=accuracy_score(test['Value'],y_pred)
print(accuracy*100)
acc.append(round(accuracy,2))

#using SVM_RBF
from sklearn.svm import SVC
clf=SVC(kernel='rbf',gamma='auto')
clf.fit(train.drop(['Value'],axis=1),train['Value'])
C1Output=clf.predict(test.drop(['Value'],axis=1))
accuracy=accuracy_score(test['Value'],C1Output)
print("Accuracy using SVM")
print(accuracy*100)
acc.append(round(accuracy,2))

#using SVM_Linear
from sklearn.svm import SVC
clf=SVC(kernel='linear',gamma='auto')
clf.fit(train.drop(['Value'],axis=1),train['Value'])
C1Output=clf.predict(test.drop(['Value'],axis=1))
accuracy=accuracy_score(test['Value'],C1Output)
print("Accuracy using SVM")
print(accuracy*100)
acc.append(round(accuracy,2))


label=['KNN','Logistic Regression','SVM_RBF','SVM_LINEAR']
index = np.arange(len(acc))
plt.bar(index, acc)
plt.xlabel('Percentage', fontsize=5)
plt.ylabel('Algorithm', fontsize=5)
plt.xticks(index, label, fontsize=5, rotation=30)
plt.title('Accuracy')
plt.show()






