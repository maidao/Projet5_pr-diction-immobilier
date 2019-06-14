import requests

url = 'https://myfirst-roku.herokuapp.com/'

params = {
    'superficie': 72,
    'nb_pieces': 3,
    'latitude': 45.7356,
    'longitude': 4.8314335
}

session = requests.Session()
r = session.get(url=url, params=params)
print(r.text)