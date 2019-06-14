from flask import request, Flask, jsonify
from class_analyse_data import *
from class_linear_predict_multi_var import *
from distance import *

app = Flask(__name__)
def predict(surface, piece, lat, lon):
    dis = compute_distance(lon,lat)
    path2017 = "69_2017.csv"
    data_2017 = Clean_data(path2017)
    d17 = data_2017.clean()
    model_2017_mul_var = Linear_predict_multi_var()
    model = model_2017_mul_var.multi_variabe_valeur_fonciere_avec_surface_et_nb_piece(d17)
    return model.predict([[surface, piece, dis]])

@app.route("/", methods=['GET', 'POST'])
def test():
    lat = float(request.args['latitude'])
    lon = float(request.args['longitude'])
    surface = float(request.args['superficie'])
    pieces = float(request.args['nb_pieces'])

    res = predict(surface, pieces, lat, lon)
    print(res)

    return jsonify({'estimation_valeur_fonciere': res.tolist()})

if __name__ == "__main__":
    app.run(debug=True)