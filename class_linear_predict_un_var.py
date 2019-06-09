from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

class Linear_predict_un_var:

    def un_variabe_surface_valeur_fonciere(self,data):

       # data.plot.scatter(x='surface_reelle_bati', y='valeur_fonciere')

        X = pd.DataFrame(data['surface_reelle_bati'])
        Y = pd.DataFrame(data['valeur_fonciere'])

        #X = data.iloc[:, 2].values.reshape(-1, 1)
        #Y = data.iloc[:,-1].values
        model = linear_model.LinearRegression()
        model.fit(X, Y)
        a = model.coef_
        b = model.intercept_
        s = model.score(X,Y)
        print("Hệ số hồi quy: a = %.2f" % a)
        print("Sai số: b = %.2f" % b)
        print("sai lech: c = %.2f" % s)

        plt.figure("data original")
        plt.scatter(X, Y, c='r')
        plt.plot(X, model.predict(X))
        #plt.show()

        print("-----------------------------")

        X_train = X[:-1000]
        X_test = X[-1000:]

        Y_train = Y[:-1000]
        Y_test = Y[-1000:]

        model_test = linear_model.LinearRegression()
        model_test.fit(X_train,Y_train) #co duoc mo hinh

        # tu mo hinh xay dung duoc voi du lieu huan luyen, ta cho tap du lieu khac (x_test) vao
        # test xem mo hinh du doan dua tren mo hinh huan luyen, cho ra ket qua (y_pred) dung voi
        # bao nhieu phan tram mo hinh that (y_test)
        Y_pred = model_test.predict(X_test)

        r = r2_score(Y_test,Y_pred)
        mse = mean_squared_error(Y_test,Y_pred)

        print("Variance score: %.2f" % r)
        print("mean_squared_error: %.2f " % mse)

        plt.figure("model test entre valeur_fonciere et surface_reelle_bati")
        plt.scatter(X_test, Y_test, c='black')
        plt.plot(X_test, Y_pred, color='blue', linewidth=3)

        plt.show()

        return a, b, s

    def un_variabe_nb_pieces_valeur_fonciere(self, data):
        #data.plot.scatter(x='nombre_pieces_principales', y='valeur_fonciere')

        X = pd.DataFrame(data['nombre_pieces_principales'])
        Y = pd.DataFrame(data['valeur_fonciere'])

        model = linear_model.LinearRegression()
        model.fit(X, Y)

        a = model.coef_
        b = model.intercept_
        s = model.score(X, Y)
        print("a = %.2f" % a)
        print("b = %.2f" % b)
        print("score = %.2f" % s)

        plt.figure("data original")
        plt.scatter(X, Y, c='r')
        plt.plot(X, model.predict(X))
        # plt.show()

        print("-----------------------------")

        X_train = X[:-1000]
        X_test = X[-1000:]

        Y_train = Y[:-1000]
        Y_test = Y[-1000:]

        model_test = linear_model.LinearRegression()
        model_test.fit(X_train, Y_train)

        Y_pred = model_test.predict(X_test)

        r = r2_score(Y_test, Y_pred)
        mse = mean_squared_error(Y_test, Y_pred)

        print("Variance score: %.2f" % r)
        print("mean_squared_error: %.2f " % mse)

        plt.figure("model test entre valeur_fonciere et nombre_pieces_principales")
        plt.scatter(X_test, Y_test, c='black')
        plt.plot(X_test, Y_pred, color='blue', linewidth=3)

        plt.show()

    def un_variabe_superficie_valeur_fonciere(self, data):
        #data.plot.scatter(x='code_postal', y='valeur_fonciere')

        X = pd.DataFrame(data['code_postal'])
        Y = pd.DataFrame(data['prix_m2'])

        model = linear_model.LinearRegression()
        model.fit(X, Y)

        a = model.coef_
        b = model.intercept_
        s = model.score(X, Y)
        print("a = %.2f" % a)
        print("b = %.2f" % b)
        print("score = %.2f" % s)

        plt.figure("data original")
        plt.scatter(X, Y, c='r')
        plt.plot(X, model.predict(X))
        # plt.show()

        print("-----------------------------")

        X_train = X[:-1000]
        X_test = X[-1000:]

        Y_train = Y[:-1000]
        Y_test = Y[-1000:]

        model_test = linear_model.LinearRegression()
        model_test.fit(X_train, Y_train)

        Y_pred = model_test.predict(X_test)

        r = r2_score(Y_test, Y_pred)
        mse = mean_squared_error(Y_test, Y_pred)

        print("Variance score: %.2f" % r)
        print("mean_squared_error: %.2f " % mse)

        plt.figure("model test entre valeur_fonciere et la superficie")
        plt.scatter(X_test, Y_test, c='black')
        plt.plot(X_test, Y_pred, color='blue', linewidth=3)

        plt.show()