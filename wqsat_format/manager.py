import os

# Wqsat Format
from wqsat_format import s2_reader, s3_reader
from wqsat_format import composite
from wqsat_format import utils

class FormatManager:
    """
    Class to manage Sentinel-2 and Sentinel-3 data processing.
    """
    def __init__(self, config: dict = None, config_file: str = None):
        if config:
            self.config = config
        elif config_file:
            self.config = self.from_yaml(config_file)
        else:
            raise ValueError("Either config or config_path must be provided.")
        
        self.settings = {
            "satellite": self.config.get("satellite"),
            "tile_path": self.config.get("tile_path"),
            "bands": self.config.get("bands", []),
            "roi_lat_lon": self.config.get("roi_lat_lon", None),
            "roi_window": self.config.get("roi_window", None),
            "atcor": self.config.get("atcor", False),
            "temporal_composite": self.config.get("temporal_composite", None),
            "spatial_composite": self.config.get("spatial_composite", False),
            "output_dir": self.config.get("output_dir", '.'),
            "output_format": self.config.get("output_format", "GeoTIFF")}
        
        ## Validate inputs
        utils.validate_inputs(self.settings)
        os.makedirs(self.settings['output_dir'], exist_ok=True)
        
    def workflow(self):
        """
        Runs the selected reader to process the data.
        """

        if isinstance(self.settings['tile_path'], str):
            #Leer y exportar una sola tile
            if self.settings['satellite'] == "SENTINEL-2":
                reader = s2_reader.S2Reader(tile_path= self.settings['tile_path'], bands=self.settings['bands'],
                                            roi_lat_lon=self.settings['roi_lat_lon'], roi_window=self.settings['roi_window'],
                                            atcor=self.settings['atcor'])
            elif self.settings['satellite'] == "SENTINEL-3":
                reader = s3_reader.S3Reader(tile_path=self.settings['tile_path'], bands=self.settings['bands'],
                                            roi_lat_lon=self.settings['roi_lat_lon'], roi_window=self.settings['roi_window'],
                                            atcor=self.settings['atcor'])
            else:
                raise ValueError("❌ Unsupported satellite type. Please use 'SENTINEL-2' or 'SENTINEL-3'.")
                                            
            # Read the bands and metadata
            band_data, metadata, output_filename = reader.read_bands()
            return band_data, metadata, output_filename
            # utils.export_data(band_data, metadata, self.output_path, output_filename, self.settings['output_format'])
        
        elif isinstance(self.settings['tile_path'], list):
            
            if not self.settings['tile_path']:
                raise ValueError("❌ The list of tile paths is empty.")
            
            if self.settings['satellite'] != "SENTINEL-2":
                raise ValueError("❌ Temporal and spatial composites are only available for Sentinel-2.")
            
            # Composing temporal or spatial tiles
            if self.settings['temporal_composite'] and self.settings['spatial_composite']:  
                raise ValueError("❌ Both temporal and spatial composites cannot be applied simultaneously.")
            
            if self.settings['temporal_composite']:
                if self.settings['temporal_composite'] not in ["median", "max"]:
                    raise ValueError("❌ Invalid composite type. Use 'median' or 'max'.")
                
                # Create a temporal composite            
                reader = composite.CompositeManager(tile_path=self.settings['tile_path'], bands=self.settings['bands'],
                                                    roi_lat_lon=self.settings['roi_lat_lon'], roi_window=self.settings['roi_window'],
                                                    atcor=self.settings['atcor'], temporal_composite=self.settings['temporal_composite'])
                composite_arr, composite_metadata, output_filename = reader.compose_temporal_tiles(self.settings['tile_path'])
                return composite_arr, composite_metadata, output_filename
                # utils.export_data(composite_arr, composite_metadata, self.output_path, output_filename)
                    
            elif self.settings['spatial_composite']:
                reader = composite.CompositeManager(tile_path=self.settings['tile_path'], bands=self.settings['bands'],
                                                    roi_lat_lon=self.settings['roi_lat_lon'], roi_window=self.settings['roi_window'],
                                                    atcor=self.settings['atcor'])
                
                composite_arr, composite_metadata, output_filename = reader.compose_spatial_tiles(self.settings['tile_path'])
                return composite_arr, composite_metadata, output_filename
                # utils.export_data(composite_arr, composite_metadata, self.output_path, output_filename)

            else:
                for tile_path in self.settings['tile_path']:
                    print(f"Reading tile {tile_path}...")
                    reader = s2_reader.S2Reader(tile_path=tile_path, bands=self.settings['bands'], roi_lat_lon=self.settings['roi_lat_lon'], 
                                                roi_window=self.settings['roi_window'], atcor=self.settings['atcor'])
                    band_data, metadata, output_filename = reader.read_bands()
                    return band_data, metadata, output_filename
                    # utils.export_data(band_data, metadata, self.output_path, output_filename)
        else:
                raise ValueError("❌ Invalid 'tile path'. Please provide a string or a list of strings.")