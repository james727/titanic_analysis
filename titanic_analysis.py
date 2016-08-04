import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.grid_search import GridSearchCV


filename = 'train.csv'
titanic_data = pd.read_csv(filename)
"""print titanic_data.head()
living = titanic_data[titanic_data['Survived']==1]
dead = titanic_data[titanic_data['Survived']==0]
living_plot = plt.scatter(living['Age'],living['Fare'],c='g',marker='o')
dead_plot = plt.scatter(dead['Age'],dead['Fare'],c='r',marker='o')
#plt.legend((living_plot,dead_plot),('Survived','Did not make it'))
#plt.show()"""

titanic_data = titanic_data[np.isfinite(titanic_data['Age'])]
labels = titanic_data['Survived'].values
del titanic_data['Survived']
del titanic_data['PassengerId']
del titanic_data['Name']
del titanic_data['Ticket']
del titanic_data['Cabin']
del titanic_data['Embarked']
titanic_data['Sex'] = titanic_data.Sex.apply(lambda x: 1 if x=='male' else 0)
features = titanic_data.as_matrix()

train_fraction = 0.5
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size = train_fraction)

# Naive bayes
clf = GaussianNB()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
print "The accuracy of Gaussian Naive Bayes is: " + str(accuracy_score(pred, labels_test))

# svm
parameters = {'kernel':('linear', 'rbf'), 'C':[.1, 1,10,100]}
svr = SVC()
clf = GridSearchCV(svr, parameters)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "The accuracy of SVM is: " + str(accuracy_score(pred, labels_test))
print "The best parameters are:"
print clf.best_params_

# Decision tree
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "The accuracy of Decision Tree is: " + str(accuracy_score(pred, labels_test))
