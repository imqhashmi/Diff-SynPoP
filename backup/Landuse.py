import pandas as pd
import os
import geopandas as gpd
from shapely.geometry import Point
import requests


# Function to get elevation
def get_elevation(lat, lon):
    url = "https://api.open-elevation.com/api/v1/lookup"
    query = {'locations': f'{lat},{lon}'}
    response = requests.get(url, params=query).json()
    elevation = response['results'][0]['elevation']
    return elevation
filepath = os.path.join(os.path.dirname(os.getcwd()),  'VE', 'GISRegion2.csv')
df = pd.read_csv(filepath)

# # Convert DataFrame to GeoDataFrame
# gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
# gdf['elevation'] = gdf.apply(lambda row: get_elevation(row['latitude'], row['longitude']), axis=1)
# gdf.to_file("output.geojson", driver='GeoJSON')
