import os
import numpy as np
import glob
import xml.etree.ElementTree as ET
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds
import xarray as xr

from wqsat_format import atcor

class S2Reader:
    def __init__(self, tile_path, bands=None, roi_lat_lon=None, roi_window=None, atcor=True, crs="EPSG:4326", output_format="GeoTIFF"):
        """
        Initializes the Sentinel-2 Reader class.

        Parameters
        ----------
        tile_path : str
            Path to the Sentinel-2 tile folder.
        bands : list of str, optional
            List of bands to read. If None, all bands are read.
        roi_lat_lon : dict, optional
            Dictionary with bounding box (W, N, E, S) in latitude/longitude.
        roi_window : list of int, optional
            List defining the window in pixels (xmin, ymin, xmax, ymax).
        atcor : bool, optional
            Whether to apply atmospheric correction (Dark object subtraction, DOS).
        crs : str, optional
            Coordinate reference system of input coordinates.
        """
        self.tile_path = tile_path
        self.roi_lat_lon = roi_lat_lon
        self.roi_window = roi_window
        self.crs = crs
        self.atcor = atcor
        self.output_format = output_format
        
        # Default to all available bands if none are specified
        self.bands = bands or ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

    def calculate_window(self, src, resolution):
        """
        Calculate the window for the given resolution.

        Parameters
        ----------
        src : rasterio.DatasetReader
            The raster dataset being processed.
        resolution : int
            The spatial resolution of the band (10, 20, 60 meters).
        
        Returns
        -------
        rasterio.windows.Window or None
            The computed window or None if no ROI is provided.
        """
        factor = int(resolution) // 10 # Adjust window size based on resolution factor

        if self.roi_lat_lon:
            # Convert geographic coordinates to image bounds
            W, N, E, S = self.roi_lat_lon['W'], self.roi_lat_lon['N'], self.roi_lat_lon['E'], self.roi_lat_lon['S']
            bounds = transform_bounds(self.crs, src.crs, float(W), float(S), float(E), float(N))
            return from_bounds(*bounds, transform=src.transform)

        elif self.roi_window:

            # Use the provided pixel window
            col_off = self.roi_window['xmin'] // factor
            row_off = self.roi_window['ymin'] // factor
            width = (self.roi_window['xmax'] - self.roi_window['xmin']) // factor
            height = (self.roi_window['ymax'] - self.roi_window['ymin']) // factor
            
            # Scale the window based on resolution factor
            return Window(col_off, row_off, width, height)
        
        return None
    
    def get_SunAngles(self):
        """
        Reads the MTD_TL.xml file and extracts the sun angles (zenith and azimuth).
        
        Returns
        -------
        'sza' (solar zenith angle) and 'saa' (solar azimuth angle).
        """
        # Locate the MTD_TL.xml file inside the GRANULE subdirectory
        granule_path = os.path.join(self.tile_path, "GRANULE")
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
        
        # Extract solar angle from the correct file
        sun_angle_node = root.find(".//Mean_Sun_Angle")
        if sun_angle_node is not None:
            sza =  float(sun_angle_node.find("ZENITH_ANGLE").text),
            saa =  float(sun_angle_node.find("AZIMUTH_ANGLE").text)
    
        return sza, saa

    def read_bands(self):
        """
        Reads the selected bands from the Sentinel-2 tile.
        
        Returns
        -------
        tuple
            - dict: Contains band data categorized by resolution.
            - dict: Metadata for each resolution.
        """

        try:
            # Locate the IMG_DATA folder within the Sentinel-2 tile structure
            granule_path = glob.glob(os.path.join(self.tile_path, "GRANULE", "*", "IMG_DATA"))[0]
            jp2_files = {os.path.basename(f).split("_")[-1].split(".")[0]: f for f in glob.glob(os.path.join(granule_path, "*.jp2"))}
        except IndexError:
            raise ValueError("Granule path not found in the Sentinel-2 tile directory.")
        
        valid_bands = [b for b in self.bands if b in jp2_files]
        if not valid_bands:
            raise ValueError("No valid bands found in the Sentinel-2 tile.")
        
        # Band groupings by resolution
        resolution_bands = {
            "10": ["B04", "B03", "B02", "B08"],
            "20": ["B05", "B06", "B07", "B8A", "B11", "B12"],
            "60": ["B01", "B09", "B10"]
        }

        # Initialize dictionaries for data and metadata
        arr_bands, metadata = {}, {}
        if self.atcor:
            sza, saa = self.get_SunAngles()

        for res, bands in resolution_bands.items():
            arr_bands[res] = {}
            bands_list = []
            for band in bands:
                if band not in valid_bands:
                    continue
                try:
                    print(f"Reading band {band} at {res}m resolution...")
                    with rasterio.open(jp2_files[band]) as src:
                        Window = self.calculate_window(src, int(res))
                        data = src.read(1, window=Window)

                        if self.atcor:
                            # Apply ATCOR corrections
                            data = atcor.Atcor(data, sza, saa).apply_all_corrections()

                        arr_bands[res][band] = data
                        bands_list.append(band)

                        # Update metadata for this resolution
                        if res not in metadata:
                            metadata[res] = src.meta.copy()
                            metadata[res].update({
                                "transform": src.window_transform(Window) if Window else src.transform,
                                "width": data.shape[1],
                                "height": data.shape[0],
                                "count": len(bands),
                                "dtype": data.dtype,
                                "bands": bands_list
                            })

                except Exception as e:
                    raise RuntimeError(f"Error reading band {band} at {res}m resolution: {e}")

        self.export_data(arr_bands, metadata)

    def export_data(self, arr_bands, metadata):

        file = os.path.basename(os.path.normpath(self.tile_path))
        file = file.split('.')[0]

        for res, meta in metadata.items():
            output_file = os.path.join(self.tile_path, f"{file}_{res}m.tif")

            # Asegurar compatibilidad con GeoTIFF y float32
            if self.output_format.lower() == "geotiff":
                meta.update({
                    "count": len(meta["bands"]),
                    "dtype": np.float32,  # Mantener float32
                    "driver": "GTiff",    # Forzar GeoTIFF
                })

                with rasterio.open(output_file, "w", **meta) as dst:
                    for i, band in enumerate(meta["bands"], start=1):
                        dst.write(arr_bands[res][band], i)

            # Guardar en NetCDF
            elif self.output_format.lower() == "netcdf":
                data_vars = {band: (("y", "x"), arr_bands[res][band]) for band in meta["bands"]}
                ds = xr.Dataset(
                    data_vars=data_vars,
                    coords={
                        "y": np.arange(meta["height"]),
                        "x": np.arange(meta["width"])
                    }
                )
                ds.to_netcdf(output_file)

            else:
                raise ValueError("Unsupported output format. Use 'GeoTIFF' or 'NetCDF'.")