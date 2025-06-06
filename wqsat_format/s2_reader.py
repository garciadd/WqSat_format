import os, re
import glob
import rasterio
import xml.etree.ElementTree as ET
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds

from wqsat_format import atcor

class S2Reader:
    def __init__(self, tile_path, bands=None, roi_lat_lon=None, roi_window=None, atcor=False, crs="EPSG:4326"):
        """
        Initializes the S2Reades class for reading Setinel-2 SAFE or GeoTIFF imagery.

        Parameters
        ----------
        tile_path (str): Path to SAFE directory or GeoTIFF file.
        bands (list of str, optional): List of bands to read. If None, all bands are read.
        roi_lat_lon (dict, optional): Geographic ROI. Dictionary with bounding box {W, N, E, S} in latitude/longitude.
        roi_window (dict, optional): Pixel ROI. Dictionary defining the window in pixels {xmin, ymin, xmax, ymax}.
        atcor (bool): Whether to apply atmospheric correction (Dark object subtraction, DOS).
        crs (str): Coordinate reference system of input coordinates. Default is "EPSG:4326".
        """ 
        # Initialize reader with path to SAFE directory or GeoTIFF file
        self.tile_path = tile_path
        if not os.path.exists(self.tile_path):
            raise FileNotFoundError(f"Tile path does not exist: {self.tile_path}")
        
        self.roi_lat_lon = roi_lat_lon
        self.roi_window = roi_window
        self.atcor = atcor
        self.crs = crs        
        self.bands = bands
        
    def get_window(self, src, resolution):
        """
        Compute rasterio Window object for cropping the image.

        Parameters:
        src (rasterio DatasetReader): Opened rasterio source.
        resolution (int): Image resolution in meters.

        Returns:
        tuple: (rasterio.windows.Window, bounds) or (None, None) if no ROI.
        """
        factor = int(resolution) // 10 # Adjust window size based on resolution factor

        if self.roi_lat_lon:
            # Convert geographic coordinates to image bounds
            W, N, E, S = self.roi_lat_lon['W'], self.roi_lat_lon['N'], self.roi_lat_lon['E'], self.roi_lat_lon['S']
            bounds = transform_bounds(self.crs, src.crs, float(W), float(S), float(E), float(N))

            # Clip the requested bounds to the image bounds
            left, bottom, right, top = src.bounds
            new_bounds = (
                max(bounds[0], left),
                max(bounds[1], bottom),
                min(bounds[2], right),
                min(bounds[3], top)
            )

            # Convert clipped bounds to pixel window
            window = from_bounds(*new_bounds, transform=src.transform)
            return window, new_bounds

        elif self.roi_window:

            # Convert pixel coordinates with resolution scaling
            col_off = self.roi_window['xmin'] // factor
            row_off = self.roi_window['ymin'] // factor
            width = (self.roi_window['xmax'] - self.roi_window['xmin']) // factor
            height = (self.roi_window['ymax'] - self.roi_window['ymin']) // factor
            
            # Scale the window based on resolution factor
            window = Window(col_off, row_off, width, height)
            return window, None
        
        return None, None
    
    def get_SunAngles(self):

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
            sza =  float(sun_angle_node.find("ZENITH_ANGLE").text)
            saa =  float(sun_angle_node.find("AZIMUTH_ANGLE").text)
    
        return sza, saa

    def read_safe(self):
        """
        Read Sentinel-2 imagery from SAFE format (.jp2) structure.

        Returns:
        dict: Band data grouped by resolution.
        dict: Metadata for each resolution.
        """
        try:
            # Locate the IMG_DATA folder within the Sentinel-2 tile structure
            granule_path = glob.glob(os.path.join(self.tile_path, "GRANULE", "*", "IMG_DATA"))[0]
            subdirs = [d for d in os.listdir(granule_path) if os.path.isdir(os.path.join(granule_path, d)) and d.startswith("R")]
            if subdirs:
                jp2_files_list = glob.glob(os.path.join(granule_path, "R*", "*.jp2"))
            else:
                jp2_files_list = glob.glob(os.path.join(granule_path, "*.jp2"))

            jp2_files = {}
            for f in jp2_files_list:
                filename = os.path.basename(f)
                match = re.search(r'_(B\d{2}|B8A|AOT|WVP|TCI|SCL)(?=\.jp2|_|$)', filename)
                if match:
                    band_id = match.group(1)
                    jp2_files[band_id] = f
        except IndexError:
            raise ValueError("Granule path not found in the Sentinel-2 tile directory.")
        
        valid_bands = [b for b in self.bands if b in jp2_files]
        valid_bands.sort(key=lambda x: self.bands.index(x))
        if not valid_bands:
            raise ValueError("No valid bands found in the Sentinel-2 tile.")
        
        # Band groupings by resolution
        resolution_bands = {
            10: ["B04", "B03", "B02", "B08"],
            20: ["B05", "B06", "B07", "B8A", "B11", "B12"],
            60: ["B01", "B09", "B10"]
        }

        # Initialize dictionaries for data and metadata
        arr_bands, metadata = {}, {}
        for res, bands in resolution_bands.items():
            selected_bands = [band for band in bands if band in valid_bands]
            if not selected_bands:
                continue

            arr_bands[res] = {}
            bands_list = []
            for band in selected_bands:
                try:
                    print(f"Reading band {band} at {res}m resolution...")
                    with rasterio.open(jp2_files[band]) as src:
                        window, new_bounds = self.get_window(src, int(res))
                        data = src.read(1, window=window)

                        if self.atcor:
                            # Apply ATCOR corrections
                            sza, saa = self.get_SunAngles()
                            data = atcor.Atcor(data, sza, saa).apply_all_corrections()  

                        arr_bands[res][band] = data
                        bands_list.append(band)

                        # Update metadata for this resolution
                        if res not in metadata:
                            meta = src.meta.copy()
                            if window:
                                meta.update({
                                    "transform": src.window_transform(window),
                                    "width": data.shape[1],
                                    "height": data.shape[0],
                                })
                            meta.update({
                                "count": len(bands_list),
                                "dtype": data.dtype,
                                "bands": bands_list,
                                "bounds": new_bounds if new_bounds else src.bounds,
                                "crs": src.crs,
                                "transform_affine": list(src.transform)[:6],
                                "resolution": src.transform[0],
                            })
                            metadata[res] = meta
                except Exception as e:
                    raise RuntimeError(f"Error reading band {band} at {res}m resolution: {e}") 
        return arr_bands, metadata
    
    def read_geotiff(self):
        """
        Read bands from a single GeoTIFF file using ROI and descriptions.

        Returns:
        dict: Band data grouped by resolution.
        dict: Metadata for the resolution.
        """
        arr_bands, metadata = {}, {}
        with rasterio.open(self.tile_path) as src:
            res = int(src.transform[0])
            window, new_bounds = self.get_window(src, res)

            band_names = [src.descriptions[i] if src.descriptions[i] else f"Band_{i+1}" for i in range(src.count)]
            available_bands = dict(zip(band_names, range(1, src.count + 1)))
            bands_to_read = [b for b in self.bands if b in available_bands]
            if not bands_to_read:
                raise ValueError("No valid bands found in the GeoTIFF file.")
            
            arr_bands[str(res)] = {}
            for band_name in bands_to_read:
                band_index = available_bands[band_name]
                data = src.read(band_index, window=window)
                arr_bands[str(res)][band_name] = data
            meta = src.meta.copy()
            if window:
                meta.update({
                    "transform": src.window_transform(window),
                    "width": data.shape[1],
                    "height": data.shape[0]
                })
            meta.update({
                "crs": src.crs,
                "bands": bands_to_read,
                "resolution": src.transform[0],
                "transform_affine": list(src.transform)[:6],
                "bounds": new_bounds if new_bounds else src.bounds
            })
            metadata[str(res)] = meta
        return arr_bands, metadata
    
    def read_bands(self):
        """
        Wrapper method to read bands from SAFE or GeoTIFF based on file type.

        Returns:
        dict: Band data grouped by resolution.
        dict: Metadata.
        """
        if self.tile_path.endswith(".SAFE"):
            band_data, metadata = self.read_safe()
        elif self.tile_path.endswith(".tif") or self.tile_path.endswith(".tiff"):
            band_data, metadata = self.read_geotiff()
        else:
            raise ValueError("Unsupported file format. Please provide a .SAFE or .tif file.")
        
        filename = os.path.basename(os.path.normpath(self.tile_path)).split('.')[0]
        return band_data, metadata, filename