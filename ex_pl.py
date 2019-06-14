import numpy as np
import matplotlib.pyplot as plt

X = np.array([0, 2, 10, 8])
Y = np.array([0, 8, 0, 4])
X_mean1 =
Y_mean1 =

X_mean2 =
Y_mean2 =

print(Y_mean1, Y_mean2)

plt.scatter(X,Y)
plt.plot()
plt.show()


for i in range(1,5):
    X = eval("df_" + str(i) + ".iloc[:,0].values.reshape(-1, 1)")
    Y = eval("df_" + str(i) + ".iloc[:,-1]")
