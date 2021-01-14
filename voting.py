from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from keras.models import load_model
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

Xi = []
Y = []

with open("/Users/admin/Desktop/e/normalized.csv", 'r') as r:
    c = 0
    reader = csv.reader(r)
    for row in reader:
        if c==0:
            c+=1
            continue

        for j in range(0,len(row)):
            row[j] = float(row[j])
        Xi.append(row)
n_models = 5
all_models=[]
#a = ('m1', 'm2', 'm3', 'm4', 'm5')
""""
for i in range(n_models):
    filename = '/Users/admin/Desktop/ensemble_neural/models/' + str(i + 1) + '.h5'
    model = load_model(filename)
    all_models.append(model)
    print('>loaded %s' % filename)
"""
train ,test = train_test_split(Xi,test_size=0.25)
X = []
Xt = []
Yt = []
acc = []
for i in train:
    X.append(i[:len(i)-1])
    Y.append(i[len(i)-1:][0])
for i in test:
    Xt.append(i[:len(i)-1])
    Yt.append(i[len(i) - 1:][0])
print(len(Y))
print(len(Yt))

logreg = LogisticRegression()
clf = LinearDiscriminantAnalysis()
clf1 = QuadraticDiscriminantAnalysis()
listt = list()
listt.append(('m1',clf))
listt.append(('m2',clf1))
listt.append(('m3',logreg))
estimator = []
estimator.append(('LR', listt[1]))
estimator.append(('LDA', listt[2]))
estimator.append(('QDA', listt[3]))
voting_clf1 = VotingClassifier(estimator, voting='hard')

# Train the model using the training sets
voting_clf1.fit(X,Y)

#Predict Output
y_pred= voting_clf1.predict(Xt)
acc = accuracy_score(y_pred, Yt)
print(acc)



"""
voting_clf = VotingClassifier(all_models, voting='hard')

# Train the model using the training sets
voting_clf.fit(X,Y)

#Predict Output
y_pred= voting_clf.predict(Xt)
acc = accuracy_score(y_pred, Yt)
print(acc)
count = 0
j = 0
for i in y_pred:
    if int(i) == int(Yt[j]):
        count = count+1
    j+=1
print(count)
print((count/len(Yt))*100)
acc.append((count/len(Yt))*100)
print("=== Confusion Matrix ===")
print(confusion_matrix(Yt, y_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(Yt, y_pred))
print('\n')
"""