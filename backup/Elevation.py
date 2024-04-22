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

    # Check if 'results' key exists
    if 'results' in response and len(response['results']) > 0:
        elevation = response['results'][0]['elevation']
        print(f"The elevation of {lat}, {lon} is {elevation} meters.")
        return elevation
    else:
        # Handle the case where 'results' is not present
        print(f"No elevation data available for {lat}, {lon}")
        return 0  # or an appropriate default value


filepath = os.path.join(os.path.dirname(os.getcwd()), 'Diff-SynPoP',  'GISRegion2.csv')
df = pd.read_csv(filepath)

# Convert DataFrame to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
gdf['elevation'] = gdf.apply(lambda row: get_elevation(row['lat'], row['lon']), axis=1)
gdf.to_file("output.geojson", driver='GeoJSON')
