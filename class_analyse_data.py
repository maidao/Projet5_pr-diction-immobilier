import pandas as pd
from list_constans import *
from distance import *


class Clean_data:
    def __init__(self, path):
        self.data_raw = pd.read_csv(path,low_memory=False)

    def clean(self):
        # select columns necessary
        data = self.data_raw[['code_postal','valeur_fonciere','nombre_pieces_principales',
                                    'surface_reelle_bati','type_local',
                                    'longitude','latitude']]

        # select rows with type and location precise
        data = data[data['type_local'].isin(type)]
        data = data[data['code_postal'].isin(lyon)]

        # remove rows NaN
        data = data[~data['surface_reelle_bati'].isnull()]
        data = data[~data['valeur_fonciere'].isnull()]
        data = data[~data['longitude'].isnull()]
        data = data[~data['latitude'].isnull()]
        data = data[~data['nombre_pieces_principales'].isnull()]

        # add column prix/m2
        data = data[(data['valeur_fonciere'] <= 1500000) & (data['valeur_fonciere'] >= 50000)]
        data = data[(data['surface_reelle_bati'] >= 17) & (data['surface_reelle_bati'] <= 250)]
        data['prix_m2'] = round(data.valeur_fonciere / data.surface_reelle_bati)
        data = data[(data['prix_m2'] >= 500) & (data['prix_m2'] <= 10000)]
        data = data[(data['nombre_pieces_principales'] <= 8)]
        data['distance_to_centre'] = compute_distance(data.longitude, data.latitude)

        return data



