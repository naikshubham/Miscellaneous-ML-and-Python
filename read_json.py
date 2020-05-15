import geojson

file = 'data/maps.geojson'

with open(file) as f:
    gj = geojson.load(f)
features = gj['features']
for dist in features:
    print(dist["properties"])