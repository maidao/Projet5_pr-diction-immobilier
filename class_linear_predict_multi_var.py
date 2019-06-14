from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class Linear_predict_multi_var:

    def multi_variabe_valeur_fonciere_avec_surface_et_nb_piece(self, data):
        # tao mot dataframe chi gom cac bien giai tich

        data_X = data[['surface_reelle_bati', 'nombre_pieces_principales','distance_to_centre']]
        data_X.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)

        X = data_X.values
        Y = data["valeur_fonciere"].values

        model = linear_model.LinearRegression()
        model.fit(X, Y)
        #print(pd.DataFrame({"Name":data_X.columns, "Coefficients":model.coef_}).sort_values(by = 'Coefficients'))
        #print('b = ',model.intercept_)
        #print('score = ', model.score(X,Y))
        return model

        #X_train = X[:-1000]
        #X_test = X[-1000:]

        #Y_train = Y[:-1000]
        #Y_test = Y[-1000:]

        #model_test = linear_model.LinearRegression()
        #model_test.fit(X_train, Y_train)

        #Y_pred = model_test.predict(X_test)

        elev = 43.5
        azim = -110
        #plot_figs(1, elev, azim, X_train, Y_train, model_test)

        #plt.show()





