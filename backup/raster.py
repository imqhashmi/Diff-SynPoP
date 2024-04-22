import argparse
import rasterio

def get_landcover_type(lon, lat, landcover_file):
    # Open the land cover dataset
    with rasterio.open(landcover_file) as dataset:
        # Convert the latitude and longitude to the dataset's projection
        lon, lat = dataset.xy(lat, lon)
        # Convert the coordinates to pixel coordinates
        py, px = dataset.index(lon, lat)
        # Read the land cover value
        land_cover_value = dataset.read(1)[py, px]

    # Map the land cover value to the classification
    landcover_mapping = {
        1: "Water",
        2: "Trees",
        4: "Flooded vegetation",
        5: "Crops",
        7: "Built Area",
        8: "Bare ground",
        9: "Snow/Ice",
        10: "Clouds",
        11: "Rangeland"
    }

    return landcover_mapping.get(land_cover_value, "Unknown")

def main():
    parser = argparse.ArgumentParser(description="Get landcover type based on coordinates")
    parser.add_argument("longitude", type=float, help="Longitude")
    parser.add_argument("latitude", type=float, help="Latitude")
    parser.add_argument("landcover_file", type=str, help="Path to Landcover TIFF file")

    args = parser.parse_args()

    # Get the landcover type for the provided coordinates
    landcover_type = get_landcover_type(args.longitude, args.latitude, args.landcover_file)

    print(f"Landcover type for coordinates ({args.longitude}, {args.latitude}): {landcover_type}")

if __name__ == "__main__":
    main()
