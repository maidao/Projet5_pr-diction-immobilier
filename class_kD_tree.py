from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus
import pydot

class Kd_tree:
    def kD_tree(self,data):
        X = data[['surface_reelle_bati', 'nombre_pieces_principales', 'distance_to_centre']]
        y = data[['valeur_fonciere']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        clf = tree.DecisionTreeRegressor(max_depth=3)  # max_depth = 3 là độ sâu của cây quyết định
        clf = clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        tree.export_graphviz(clf, out_file="tree.dot",
                             feature_names= X,
                             class_names= y,
                             filled=True, rounded=True)


"""
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.3)

# tạo model
clf = tree.DecisionTreeClassifier(max_depth=3)#max_depth = 3 là độ sâu của cây quyết định
clf = clf.fit(X_train, y_train)

# Thực hiện dự đoán kết quả từ những thông số biến giải thích đầu vào
predicted = clf.predict(X_test)


Chúng ta có thể tạo cây quyết định dưới dạng file DOT. 
Và từ file DOT sẽ sử dụng GraphViz hoặc webgraphviz để mở.
Như ví dụ sau, tên của các biến giải thích được setting như sau 
feature_names=iris.feature_names, 
tên của biến mục đích class_names=iris.target_names, 
muốn màu mè cho các nhánh thì setting filled=True, 
các góc của các nhánh bo tròn cho hoành tráng thì rounded=True.


tree.export_graphviz(clf, out_file="tree.dot",
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True)
"""


# Bạn sẽ tạo được file .dot, có thể dùng trang http://www.webgraphviz.com/ để convert sang ảnh

# Hoặc dùng package pydotplus convert sang pdf, png tiện lợi hơn.
# Chưa có thì có thể dễ dàng install với pip.

#dot_data = tree.export_graphviz(clf, out_file = None , filled = True ,
                                #rounded = True , special_characters = True)
#graph = pydot.graph_from_dot_data(dot_data)
#graph.write_pdf("graph.pdf")
