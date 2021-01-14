from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import csv
import numpy as np
import matplotlib.pyplot as plt
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

n_estimators = [1, 2, 3, 4, 5, 10, 15, 20]
acc = []
for estimator in n_estimators:
    model = KNeighborsClassifier(n_neighbors = estimator)
    # Fit on training data
    model.fit(X, Y)
    # Prediction
    predicted = model.predict(Xt)
    count = 0
    j = 0
    for i in predicted:
        if int(i) == int(Yt[j]):
            count = count+1
        j+=1
    print(count)
    print((count/len(Yt))*100)
    acc.append((count/len(Yt))*100)
    print("=== Confusion Matrix ===")
    print(confusion_matrix(Yt, predicted))
    print('\n')
    print("=== Classification Report ===")
    print(classification_report(Yt, predicted))
    print('\n')

tree_list = np.array(n_estimators)
accuracy_percent = np.array(acc)
plt.plot(tree_list,accuracy_percent)
plt.xlabel('Number of neighbours')
plt.ylabel('Percent of accuracy')
plt.title('Varation of accuracy with neighbours')
plt.grid(True)
plt.savefig("knn1.png")
plt.show()
