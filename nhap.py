import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()
X = iris.data[:, :2]
iris_X = iris.data
iris_y = iris.target
print(X)
print(iris_X)

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)

# xet mot diem du lieu gan nhat (K=1)
# khoảng cách ở đây được tính là khoảng cách theo norm 2 (p=2)

model = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 1, weights = 'distance')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#print("Print results for 20 test data points:")
#print("Predicted labels: ", y_pred[20:40]) # dau ra voi mo hinh KNN
#print("Ground truth    : ", y_test[20:40]) # dau ra that su cua du lieu test
#print("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))

#print(iris_X)
#print(iris_y)