from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn import svm
from sklearn.pipeline import make_pipeline
import csv
from sklearn.metrics import classification_report, confusion_matrix
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
#Create a Gaussian Classifier
#model = GaussianNB()
level0 = list()
level0.append(('lr', RandomForestClassifier(n_estimators=60)))
level0.append(('knn', RandomForestClassifier(n_estimators=60)))
level0.append(('cart', RandomForestClassifier(n_estimators=60)))
level0.append(('svm', RandomForestClassifier(n_estimators=60)))
level0.append(('bayes', RandomForestClassifier(n_estimators=60)))
# define meta learner model
level1 = RandomForestClassifier(n_estimators=60)
# define the stacking ensemble
model = StackingClassifier(estimators=level0, final_estimator=level1)
# Train the model using the training sets
model.fit(X,Y)

#Predict Output
y_pred= model.predict(Xt)
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