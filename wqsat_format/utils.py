import os
import yaml
import xml.etree.ElementTree as ET
import numpy as np
import rasterio

def base_dir():
    """Returns the base directory where config.yaml is stored."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def config_path():
    """Returns the full path of the config.yaml file."""
    return os.path.join(base_dir(), 'config.yaml')

def regions_path():
    """Returns the full path of the regions.yaml file."""
    return os.path.join(base_dir(), 'regions.yaml')

def load_data_path():
    """
    Loads and returns the data path from the config.yaml file. Creates the directory if it doesn't exist.
    Raises an error if the directory cannot be created.
    """
    try:
        with open(config_path(), 'r') as file:
            data = yaml.safe_load(file)
            data_path = data.get('data_path', None)

            # Ensure the directory exists
            if data_path:
                if not os.path.exists(data_path):
                    try:
                        os.makedirs(data_path)
                        print(f"Directory '{data_path}' created.")
                    except Exception as e:
                        raise OSError(f"Failed to create the directory '{data_path}': {e}")

            return data_path
        
    except FileNotFoundError:
        print("Error: 'config.yaml' file not found.")
        return None
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")
    
def read_sun_angle(safe_path):
    """
    Reads the MTD_TL.xml file from a Sentinel-2 image and returns the Sun Angle
    
    Parameters
    ----------
    safe_path : str
        Path to the .SAFE directory of the Sentinel-2 image.
    
    Returns
    -------
    dict
        Dictionary containing the image Sun Angle.
    """
    # Locate the MTD_TL.xml file inside the GRANULE subdirectory
    granule_path = os.path.join(safe_path, "GRANULE")
    if not os.path.exists(granule_path):
        raise FileNotFoundError(f"Granule directory not found: {granule_path}")
    
    granule_dirs = [d for d in os.listdir(granule_path) if os.path.isdir(os.path.join(granule_path, d))]
    if not granule_dirs:
        raise FileNotFoundError("No granule subdirectory found inside GRANULE folder.")
    
    metadata_path = os.path.join(granule_path, granule_dirs[0], "MTD_TL.xml")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Parse the XML
    tree = ET.parse(metadata_path)
    root = tree.getroot()

    SunAngle = {}
    
    # Extract solar angle from the correct file
    sun_angle_node = root.find(".//Mean_Sun_Angle")
    if sun_angle_node is not None:
        SunAngle = {
            "ZENITH_ANGLE": float(sun_angle_node.find("ZENITH_ANGLE").text),
            "AZIMUTH_ANGLE": float(sun_angle_node.find("AZIMUTH_ANGLE").text)
        }
    
    return SunAngle

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

def export_by_resolution(data_bands, metadata, output_dir):
    """
    Exports the data bands grouped by resolution to GeoTIFF format.
    
    Parameters:
        data_bands (dict): A dictionary containing data for each resolution ('10', '20', '60').
                           Each resolution contains a dictionary of bands.
        metadata (dict): A dictionary containing metadata for each resolution.
        output_dir (str): The directory where the exported files will be saved.

    Returns:
        None
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each resolution
        for resolution, bands_dict in data_bands.items():
            output_file = os.path.join(output_dir, f"sentinel2_res_{resolution}m.tif")
            
            # Stack all bands into a 3D array (bands, height, width)
            list_bands = sorted(bands_dict.keys())  # Ensure band order consistency
            stacked_bands = np.stack([bands_dict[band] for band in list_bands], axis=0)

            # Get metadata for this resolution
            meta = metadata[resolution].copy()
            meta.update({
                "count": stacked_bands.shape[0],  # Number of bands
                "dtype": stacked_bands.dtype,
                "bands": list_bands
            })

            # Write to GeoTIFF
            with rasterio.open(output_file, 'w', driver='GTiff',
                               height=stacked_bands.shape[1], width=stacked_bands.shape[2],
                               count=stacked_bands.shape[0], dtype=stacked_bands.dtype,
                               crs=meta["crs"], transform=meta["transform"]) as dst:
                for band_idx in range(stacked_bands.shape[0]):
                    dst.write(stacked_bands[band_idx, :, :], band_idx + 1)

            # Save metadata to a separate text file
            metadata_file = os.path.join(output_dir, f"metadata_res_{resolution}m.txt")
            with open(metadata_file, 'w') as f:
                f.write(f"Metadata for resolution {resolution}m:\n")
                for key, value in meta.items():
                    f.write(f"{key}: {value}\n")

    except Exception as e:
        raise RuntimeError(f"Error exporting bands: {e}")