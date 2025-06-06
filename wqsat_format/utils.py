import os
import yaml
import numpy as np
import rasterio
import xarray as xr

def base_dir():
    """Returns the base directory where config.yaml is stored."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def regions_path():
    """Returns the full path of the regions.yaml file."""
    return os.path.join(base_dir(), 'regions.yaml')
    
def adjust_contrast_and_scale(band, lower_percentile=2, upper_percentile=98):
    """
    Adjusts the contrast of a band by clipping extreme values 
    (percentiles) and scaling the values between 0 and 255.
    
    Parameters:
    - band (numpy array): Sentinel-2 band.
    
    Returns:
    - numpy array: Adjusted band.
    """
    # Get the percentiles
    lower, upper = np.percentile(band, [lower_percentile, upper_percentile])

    # Clip values outside the percentile range
    band_clipped = np.clip(band, lower, upper)

    # Scale the band between 0 and 255
    band_scaled = 255 * (band_clipped - lower) / (upper - lower)
    
    return band_scaled.astype(np.uint8)

def rgb_image(band_red, band_green, band_blue):
    """
    Creates an RGB image from three Sentinel-2 bands, adjusting contrast and scaling the values 
    between 0 and 255.
    
    Parameters:
    - band_red (numpy array): Red spectrum band.
    - band_green (numpy array): Green spectrum band.
    - band_blue (numpy array): Blue spectrum band.
    - lower_percentile (float): Lower percentile for clipping values (default 2%).
    - upper_percentile (float): Upper percentile for clipping values (default 98%).
    
    Returns:
    - numpy array: RGB image in shape (height, width, 3).
    """

    # Adjusts the contrast of a band by clipping extreme values
    band_red_adjusted = adjust_contrast_and_scale(band_red)
    band_green_adjusted = adjust_contrast_and_scale(band_green)
    band_blue_adjusted = adjust_contrast_and_scale(band_blue)

    rgb_image = np.stack((band_red_adjusted, band_green_adjusted, band_blue_adjusted), axis=-1)    
    return rgb_image

def export_data(arr_bands, metadata, output_path, filename, output_format="GeoTIFF"):

    if isinstance(arr_bands, dict):
        for res, meta in metadata.items():
            meta.update({
                "count": len(meta["bands"]),
                "dtype": np.float32
            })
            if output_format.lower() == "geotiff":
                output_file = os.path.join(output_path, f"{filename}_{res}m.tif")
                meta.update({"driver": "GTiff"})
                with rasterio.open(output_file, "w", **meta) as dst:
                    for i, band in enumerate(meta["bands"], start=1):
                        dst.write(arr_bands[res][band], i)
                        dst.set_band_description(i, band)

            elif output_format.lower() == "netcdf":
                output_file = os.path.join(output_path, f"{filename}_{res}m.nc")
                transform = meta["transform"]
                height, width = meta["height"], meta["width"]
                x_coords = np.array([transform * (col, 0) for col in range(width)])[:, 0]
                y_coords = np.array([transform * (0, row) for row in range(height)])[:, 1]
                data_vars = {band: (("y", "x"), arr_bands[res][band]) for band in meta["bands"]}
                ds = xr.Dataset(
                    data_vars=data_vars,
                    coords={
                        "x": ("x", x_coords),
                        "y": ("y", y_coords)
                    },
                    attrs={
                        "crs": meta["crs"],
                        "transform": transform,
                        "resolution": res,
                        "description": f"Sentinel-2 data for {filename} at {res}m resolution",
                        "date_range": f"{meta.get('start_date', 'unknown')} to {meta.get('end_date', 'unknown')}"
                    }
                )
                ds.rio.write_crs(meta["crs"], inplace=True)
                ds.to_netcdf(output_file)
            else:
                raise ValueError("Unsupported output format. Use 'GeoTIFF' or 'NetCDF'.")
    else:
        metadata.update({
            "count": len(metadata["bands"]),
            "dtype": np.float32
        })

        ext = "tif" if output_format.lower() == "geotiff" else "nc"
        output_file = os.path.join(output_path, f"{filename}.{ext}")

        if output_format.lower() == "geotiff":
            metadata.update({"driver": "GTiff"})
            with rasterio.open(output_file, "w", **metadata) as dst:
                for i, band in enumerate(metadata["bands"], start=1):
                    band_index = i-1
                    dst.write(arr_bands[band_index], i)
                    dst.set_band_description(i, band)
        
        elif output_format.lower() == "netcdf":
            data_vars = {band: (("y", "x"), arr_bands[band]) for band in metadata["bands"]}
            transform = metadata["transform"]
            height, width = metadata["height"], metadata["width"]
            x_coords = np.array([transform * (col, 0) for col in range(width)])[:, 0]
            y_coords = np.array([transform * (0, row) for row in range(height)])[:, 1]
            ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "x": ("x", x_coords),
                    "y": ("y", y_coords)
                },
                attrs={
                    "crs": metadata["crs"],
                    "transform": transform,
                    "description": f"Sentinel-2 data for {filename}",
                    "date_range": f"{metadata.get('start_date', 'unknown')} to {metadata.get('end_date', 'unknown')}"
                }
            )
            ds.rio.write_crs(metadata["crs"], inplace=True)
            ds.to_netcdf(output_file)
        else:
            raise ValueError("Unsupported output format. Use 'GeoTIFF' or 'NetCDF'.")
    
def validate_inputs(config):

    required_fields = {
        "satellite": str,
        "tile_path": (str, list),
        "bands": list,
        "roi_lat_lon": (dict, type(None)),
        "roi_window": (dict, type(None)),
        "atcor": bool,
        "temporal_composite": (str, type(None)),
        "spatial_composite": bool
    }

    for key, expected_type in required_fields.items():
        if key not in config:
            raise ValueError(f"❌ Missing required config key: '{key}'")
        if not isinstance(config[key], expected_type):
            raise ValueError(f"❌ Invalid type for '{key}': expected {expected_type}, got {type(config[key])}")

    valid_satellites = ["SENTINEL-2", "SENTINEL-3"]
    satellite = config["satellite"]
    if satellite not in valid_satellites:
        raise ValueError(f"❌ Invalid satellite: {satellite}. Must be one of {valid_satellites}")

    s2_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 
                'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    s3_bands = ['Oa01', 'Oa02', 'Oa03', 'Oa04', 'Oa05', 'Oa06', 'Oa07',
                'Oa08', 'Oa09', 'Oa10', 'Oa11', 'Oa12', 'Oa13', 'Oa14',
                'Oa15', 'Oa16', 'Oa17', 'Oa18', 'Oa19', 'Oa20', 'Oa21']

    allowed_bands = s2_bands if satellite == "SENTINEL-2" else s3_bands
    if "bands" not in config or not config["bands"]:
        config["bands"] = allowed_bands.copy()  # Default to all bands if not specified
    for b in config["bands"]:
        if b not in allowed_bands:
            raise ValueError(f"❌ Invalid band '{b}' for {satellite}. Allowed bands: {allowed_bands}")

    if config["roi_lat_lon"]:
        if len(config["roi_lat_lon"]) != 4:
            raise ValueError("❌ 'roi_lat_lon' must be a list of four values: [W, S, E, N]")
        
        W, S, E, N = config["roi_lat_lon"]['W'], config["roi_lat_lon"]['S'], config["roi_lat_lon"]['E'], config["roi_lat_lon"]['N']
        if not (-180 <= W < E <= 180):
            raise ValueError("❌ Invalid longitude bounds: must satisfy -180 ≤ W < E ≤ 180")
        if not (-90 <= S < N <= 90):
            raise ValueError("❌ Invalid latitude bounds: must satisfy -90 ≤ S < N ≤ 90")

    if config["roi_window"]:
        if len(config["roi_window"]) != 4:
            raise ValueError("❌ 'roi_window' must be a list of four values: [xmin, ymin, xmax, ymax]")

        xmin, xmax = config["roi_window"]['xmin'], config["roi_window"]['xmax']
        ymin, ymax = config["roi_window"]['ymin'], config["roi_window"]['ymax']
        if not (xmin < xmax and ymin < ymax):
            raise ValueError("❌ 'roi_window' must satisfy: xmin < xmax and ymin < ymax")

    if config["temporal_composite"] is not None:
        valid_temporal = ["median", "max"]
        if config["temporal_composite"] not in valid_temporal:
            raise ValueError(f"❌ Invalid value for 'temporal_composite': {config['temporal_composite']}. "
                            f"Allowed values are: {valid_temporal}")

    return True