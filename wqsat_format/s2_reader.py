import os
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds
import numpy as np
import glob

class S2Reader:
    def __init__(self, tile_path, bands=None, roi_lat_lon=None, roi_window=None, crs="EPSG:4326"):
        """
        Parameters
        ----------
        tile_path: str. The path to the Sentinel-2 tile folder.
        bands: list of str. The bands to read. If None, read all bands.
        roi_lat_lon: list of float. The lat/lon of the ROI.
        roi_window: list of int. The window of the ROI.
        crs: str. The coordinate reference system of the input coordinates.
        """
        self.tile_path = tile_path
        self.roi_lat_lon = roi_lat_lon
        self.roi_window = roi_window
        self.crs = crs
        
        if bands is None:
            bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        self.bands = bands

    def calculate_window(self, src, resolution):
        """
        Calculate the window for the given resolution.

        Parameters
        ----------
        resolution: int. The resolution of the bands (10, 20, 60).

        Returns
        -------
        window: rasterio.windows.Window. The window for the given resolution.
        """
        factor = int(resolution) // 10

        if self.roi_lat_lon:
            W, N, E, S = self.roi_lat_lon['W'], self.roi_lat_lon['N'], self.roi_lat_lon['E'], self.roi_lat_lon['S']
            bounds = transform_bounds(self.crs, src.crs, float(W), float(S), float(E), float(N))
            window = from_bounds(*bounds, transform=src.transform)

        elif self.roi_window:
            window = Window(self.roi_window[0] // factor, self.roi_window[1] // factor,
                            self.roi_window[2] // factor, self.roi_window[3] // factor)
        else:
            window = None

        return window

    def read_bands(self):
        """
        Read the bands of the Sentinel-2 tile.

        Parameters
        ----------
        bands: list of str, optional
            The bands to read. If None, read all bands.
        
        Returns
        -------
        bands: dict. The bands of the Sentinel-2 tile.

        Raises
        ------
        ValueError
            If no valid bands are found in the Sentinel-2 tile.
        FileNotFoundError
            If the granule path or JP2 files are not found.
        """
        try:
            # Locate JP2 files inside the .SAFE directory
            granule_path = glob.glob(os.path.join(self.tile_path, "GRANULE", "*", "IMG_DATA"))[0]
            jp2_files = {os.path.basename(f).split("_")[-1].split(".")[0]: f for f in glob.glob(os.path.join(granule_path, "*.jp2"))}
        except IndexError:
            raise ValueError("Granule path not found in the Sentinel-2 tile directory.")
        
        valid_bands = [b for b in self.bands if b in jp2_files]
        if not valid_bands:
            raise ValueError("No valid bands found in the Sentinel-2 tile.")
        
        resolution_bands = {
            "10": ["B04", "B03", "B02", "B08"],
            "20": ["B05", "B06", "B07", "B8A", "B11", "B12"],
            "60": ["B01", "B09", "B10"]
        }

        # Initialize dictionaries for data and metadata
        arr_bands = {}
        metadata = {}

        for resolution, bands in resolution_bands.items():
            bands_list = []
            for band in bands:
                if band not in valid_bands:
                    continue
                try:
                    print(f"Reading band {band} at {resolution}m resolution...")
                    with rasterio.open(jp2_files[band]) as src:
                        Window = self.calculate_window(src, int(resolution))
                        data = src.read(1, window=Window)
                        if resolution not in arr_bands:
                            arr_bands[resolution] = {}
                        arr_bands[resolution][band] = data
                        bands_list.append(band)

                        # Update metadata for this resolution
                        if resolution not in metadata:
                            metadata[resolution] = src.meta.copy()
                            metadata[resolution].update({
                                "transform": src.window_transform(Window) if Window else src.transform,
                                "width": data.shape[1],
                                "height": data.shape[0],
                                "count": len(bands),
                                "dtype": data.dtype
                            })
                    metadata[resolution]["bands"] = bands_list  # Update bands list
                except Exception as e:
                    raise RuntimeError(f"Error reading band {band} at {resolution}m resolution: {e}")

        return arr_bands, metadata