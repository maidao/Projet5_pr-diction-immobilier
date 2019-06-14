import pandas as pd
from class_analyse_data import *
from class_linear_predict_un_var import *
from class_linear_predict_multi_var import *
from class_KNN import *
from class_kD_tree import *

path2017 = "69_2017.csv"
data_2017 = Clean_data(path2017)
d17 = data_2017.clean()
#print(len(d17))

model_2017_un_var = Linear_predict_un_var()
#print(model_2017_un_var.un_variabe_surface_valeur_fonciere(d17))

model_2017_mul_var = Linear_predict_multi_var()
print(model_2017_mul_var.multi_variabe_valeur_fonciere_avec_surface_et_nb_piece(d17))

model_knn = K_nearest_neighbor()
#print(model_knn.kNN_regression(d17))
#print(model_knn.kNN_classification(d17))

#model_kdtree = Kd_tree()
#print(model_kdtree.kD_tree(d17))


