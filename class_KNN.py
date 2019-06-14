from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import mglearn



class K_nearest_neighbor:
    def kNN_classification(self,data):
        X = data[['surface_reelle_bati', 'nombre_pieces_principales']]
        y = data[['code_postal']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50)
        neighbors_settings = range(1, 11)
        training_accuracy = []
        test_accuracy = []
        y_pred_vs_y_test = []

        for n_neighbors in neighbors_settings:

            model = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, p=2, weights = 'distance')
            model.fit(X_train, y_train)
            training_accuracy.append(model.score(X_train, y_train))
            test_accuracy.append(model.score(X_test, y_test))
            y_pred = model.predict(X_test)
            y_pred_vs_y_test.append(accuracy_score(y_test,y_pred))

        print(training_accuracy)
        print(test_accuracy)
        fig, ax = plt.subplots()
        plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
        plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
        #plt.plot(neighbors_settings, y_pred_vs_y_test, label="y_pred_vs_y_test accuracy")
        ax.set_title('kNN_classification')
        ax.set_xlabel('K - neighbor')
        ax.set_ylabel('accuracy')
        plt.legend()
        #plt.show()

        h = .02
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        for weights in ['uniform', 'distance']:
            # we create an instance of Neighbours Classifier and fit the data.
            clf = neighbors.KNeighborsClassifier(n_neighbors = 15, weights=weights)
            clf.fit(X, y)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            x_min, x_max = X.iloc[:,0].min() - 1, X.iloc[:,0].max() + 1
            y_min, y_max = X.iloc[:,1].min() - 1, X.iloc[:,1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.figure()
            plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

            # Plot also the training points
            plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y, cmap=cmap_bold,
                        edgecolor='k', s=20)
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.title("3-Class classification (k = %i, weights = '%s')"
                      % (n_neighbors, weights))

        plt.show()

    def kNN_regression(self,data):
        X = data[['surface_reelle_bati', 'nombre_pieces_principales', 'distance_to_centre']]
        y = data[['valeur_fonciere']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20)
        neighbors_settings = range(1, 11)
        training_accuracy = []
        test_accuracy = []
        y_pred_vs_y_test = []

        for n_neighbors in neighbors_settings:
            model = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, p=2, weights='distance')
            model.fit(X_train, y_train)
            training_accuracy.append(model.score(X_train, y_train))
            test_accuracy.append(model.score(X_test, y_test))
            y_pred = model.predict(X_test)
            #y_pred_vs_y_test.append(accuracy_score(y_test, y_pred))

        print(training_accuracy)
        print(test_accuracy)
        fig1, ax = plt.subplots()
        plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
        plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
        # plt.plot(neighbors_settings, y_pred_vs_y_test, label="y_pred_vs_y_test accuracy")
        ax.set_title('kNN_regression')
        ax.set_xlabel('K - neighbor')
        ax.set_ylabel('accuracy')
        plt.legend()


        fig2, axes = plt.subplots(1, 3, figsize=(15, 4))

        # khởi tạo 1000 điểm dữ liệu, cách đều nhau trong khoảng -3 và 3
        line = np.linspace(-3, 3, 1000).reshape(-1, 1)
        plt.suptitle("nearest_neighbor_regression")


        for n_neighbors, ax in zip([1, 3, 9], axes):
            reg = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors).fit(X, y)
            ax.plot(X, y, 'o')
            ax.plot(X, -3 * np.ones(len(X)), 'o')
            ax.plot(line, reg.predict(line))
            ax.set_title("%d neighbor(s)" % n_neighbors)

        plt.show()


        for i, weights in enumerate(['uniform', 'distance']):
            knn = neighbors.KNeighborsRegressor(n_neighbors=5, weights=weights)
            Y_pred = knn.fit(X, y).predict(X_test)

            plt.subplot(2, 1, i + 1)
            plt.scatter(X, y, c='k', label='data')
            plt.plot(X_test, Y_pred, c='g', label='prediction')
            plt.axis('tight')
            plt.legend()
            plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                        weights))

        plt.tight_layout()
        plt.show()









