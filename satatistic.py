from pandas.plotting import scatter_matrix
from sklearn.datasets import load_boston
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

boston = load_boston()
print(type(boston))
print(boston)
print(boston.data.shape)
print(boston.keys())
#print(boston.DESCR)
bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names
bos['PRICE'] = boston.target
print(bos.head())
print(bos.describe())
#bos.hist()
#scatter_matrix(bos, figsize=(20,20))
plt.scatter(bos['RM'],bos['PRICE'])
#plt.plot(bos['RM'],bos['PRICE'])
#plt.show()

X = bos.drop('PRICE', axis = 1)
Y = bos['PRICE']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 5)
#X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size = 0.3)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# linear Regression
lm = LinearRegression()
lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
r = r2_score(Y_test,Y_pred)
score1 = lm.score(X_train,Y_train)
score2 =lm.score(X_test,Y_test)
print("1____linear regression___")
print("mean_squared_error", mse)
print("r2 score", r)
print("score_train",score1)
print("score_test",score2)
plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
#plt.show()


print("--------------------------------------------------------------------------")
print("2____tree model____")
# Tree
clf = DecisionTreeRegressor(max_depth=3)#max_depth = 3 là độ sâu của cây quyết định
clf = clf.fit(X_train, Y_train)

# Thực hiện dự đoán kết quả từ những thông số biến giải thích đầu vào
predicted = clf.predict(X_test)
score = cross_val_score(clf, X, Y,cv = 5)
print('Score de validation croisee', score.mean())
print('score_train',clf.score(X_train,Y_train))
print('score_test',clf.score(X_test,Y_test))

tree.export_graphviz(clf, out_file="tree_boston.dot",
                         feature_names=boston.feature_names,
                         class_names=boston.target,
                         filled=True, rounded=True)


