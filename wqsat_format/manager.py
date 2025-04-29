import os, re

from wqsat_format import s2_reader
from wqsat_format import s3_reader

class SentinelWorkflowManager:
    """
    Class to manage Sentinel-2 and Sentinel-3 data processing.
    """

    def __init__(self, tile_path, bands=None, roi_lat_lon=None, roi_window=None, atcor=True, 
                 temporal_composite=None, spatial_composite=False, output_format="GeoTIFF"):
        """
        Initializes the Sentinel Workflow Manager.

        Parameters
        ----------
        tile_path : str
            Path to the Sentinel tile folder.
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
        output_format : str, optional
            Output format for the exported data. Default is "GeoTIFF".
        """
        self.tile_path = tile_path
        self.bands = bands
        self.roi_lat_lon = roi_lat_lon
        self.roi_window = roi_window
        self.atcor = atcor
        self.temporal_composite = temporal_composite
        self.spatial_composite = spatial_composite
        self.output_format = output_format

    def select_reader(self, path):
        """
        Selects the appropriate reader based on the tile path.

        Returns
        -------
        object
            An instance of the selected reader class.
        """
        tile_name = os.path.basename(path)
        if tile_name.startswith("S2"):
            return s2_reader.S2Reader(tile_path=self.tile_path, 
                                      bands=self.bands, 
                                      roi_lat_lon=self.roi_lat_lon, 
                                      roi_window=self.roi_window, 
                                      atcor=self.atcor, 
                                      temporal_composite=self.temporal_composite,
                                      output_format=self.output_format)
        elif tile_name.startswith("S3"):
            return s3_reader.S3Reader(tile_path=self.tile_path, 
                                      bands=self.bands, 
                                      roi_lat_lon=self.roi_lat_lon, 
                                      roi_window=self.roi_window, 
                                      atcor=self.atcor, 
                                      temporal_composite=self.temporal_composite,
                                      output_format=self.output_format)
        else:
            raise ValueError("Unsupported tile type. Please provide a valid Sentinel-2 or Sentinel-3 tile path.")
        
    def run(self):
        """
        Runs the selected reader to process the data.
        """
        if isinstance(self.tile_path, str):
            #Leer y exportar una sola tile
            reader = self.select_reader(self.tile_path)
            band_data, metadata = reader.read_bands()
            reader.export_data(band_data, metadata)
        
        elif isinstance(self.tile_path, list):
            if not self.tile_path:
                raise ValueError("The list of tile paths is empty.")
            
            if self.temporal_composite is not None:
                prefixes = {os.path.basename(p)[:2] for p in self.tile_path}
                if len(prefixes) > 1:
                    raise ValueError("Temporal composite only supports tiles from the same satellite type (S2 or S3).")
                print("Composing temporal tiles...")
                reader = self.select_reader(self.tile_path[0])
                reader.compose_temporal_tiles(self.tile_path)
            
            elif self.spatial_composite:
                print("Composing spatial tiles...")
                reader = self.select_reader(self.tile_path[0])
                reader.compose_spatial_tiles(self.tile_path)

            else:
                for path in self.tile_path:
                    reader = self.select_reader(path)
                    band_data, metadata = reader.read_bands()
                    reader.export_data(band_data, metadata)
        else:
                raise ValueError("Invalid tile path. Please provide a string or a list of strings.")