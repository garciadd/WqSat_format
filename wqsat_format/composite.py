import os, re
import numpy as np
from collections import defaultdict
from datetime import datetime
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.transform import array_bounds

from wqsat_format import s2_reader

class CompositeManager:
    def __init__(self, tile_path, bands=None, roi_lat_lon=None, roi_window=None, atcor=False, temporal_composite=None):
        """
        Initializes the CompositeManager class for managing composite operations.

        Parameters
        ----------
        tile_path (str or list of str): Path to SAFE directory or list of GeoTIFF files.
        bands (list of str, optional): List of bands to read. If None, all bands are read.
        roi_lat_lon (dict, optional): Geographic ROI. Dictionary with bounding box {W, N, E, S} in latitude/longitude.
        roi_window (dict, optional): Pixel ROI. Dictionary defining the window in pixels {xmin, ymin, xmax, ymax}.
        atcor (bool): Whether to apply atmospheric correction (Dark object subtraction, DOS).
        temporal_composite (bool): Whether to create a temporal composite.
        """
        self.tile_path = tile_path
        self.bands = bands
        self.roi_lat_lon = roi_lat_lon
        self.roi_window = roi_window
        self.atcor = atcor
        self.temporal_composite = temporal_composite

    def compose_temporal_tiles(self, tile_paths: list[str]):

        grid_codes = {re.search(r"T\d{2}[A-Z]{3}", os.path.basename(path)).group() for path in tile_paths}
        if len(grid_codes) != 1:
            raise ValueError(f"Tiles do not share the same grid: {grid_codes}")
        print(f"Creating temporal composite for {len(tile_paths)} tiles...")

        tile_data = []
        for path in tile_paths:
            print(f"Reading tile {os.path.basename(os.path.normpath(path))}...")
            date_str = re.search(r"\d{8}T\d{6}", path).group()[:8]
            date = datetime.strptime(date_str, "%Y%m%d")

            reader = s2_reader.S2Reader(tile_path=path, bands=self.bands, roi_lat_lon=self.roi_lat_lon, 
                                        roi_window=self.roi_window, atcor=self.atcor)
            band_data, metadata, filename = reader.read_bands()
            tile_data.append((date, band_data, metadata))

        tile_data.sort(key=lambda x: x[0])
        grouped_bands = defaultdict(lambda: defaultdict(list))

        for date, band_data, _ in tile_data:
            for res, bands in band_data.items():
                for band, arr in bands.items():
                    grouped_bands[res][band].append(arr)

        final_composite = defaultdict(dict)
        composite_metadata = {}

        for res, band_group in grouped_bands.items():
            for band, arr_list in band_group.items():
                stack = np.stack(arr_list, axis=0)
                if self.temporal_composite == "median":
                    result = np.nanmedian(stack, axis=0)
                else:
                    result = np.nanmax(stack, axis=0)
                final_composite[res][band] = result.astype(np.float32)
            _, _, metadata = tile_data[0]
            composite_metadata[res] = {
                'dtype': np.float32,
                'count': len(band_group),
                'crs': metadata[res]['crs'],
                'transform': metadata[res]['transform'],
                'height': metadata[res]['height'],
                'width': metadata[res]['width'],
                'bands': list(band_group.keys()),
                'start_date': tile_data[0][0].strftime("%Y-%m-%d"),
                'end_date': tile_data[-1][0].strftime("%Y-%m-%d")
            }

        first_date = tile_data[0][0].strftime("%Y%m%d")
        last_date = tile_data[-1][0].strftime("%Y%m%d")
        grid = next(iter(grid_codes))
        prefix = f"{self.temporal_composite.capitalize()}_temporal_composite_{grid}_{first_date}_{last_date}"
        return final_composite, composite_metadata, prefix
    
    def compose_spatial_tiles(self, tile_paths: list[str]):
        if len(tile_paths) < 2:
            raise ValueError("At least two tile paths are required for spatial composition.")
    
        date_codes = sorted([re.search(r"\d{8}T\d{6}", os.path.basename(path)).group()[:8] for path in tile_paths])
        grid_codes = {re.search(r"T\d{2}[A-Z]{3}", os.path.basename(path)).group() for path in tile_paths}
        if len(grid_codes) == 1:
            raise ValueError(f"Tiles share the same grid: {grid_codes}")

        print(f"Creating spatial composite for {len(tile_paths)} tiles...")

        all_data = defaultdict(lambda: defaultdict(list))
        bounds_list = []
        src_crs = None
        resolutions = {}
        for path in tile_paths:
            self.tile_path = path
            print(f"Reading tile {os.path.basename(os.path.normpath(self.tile_path))}...")
            reader = s2_reader.S2Reader(tile_path=path, bands=self.bands, roi_lat_lon=self.roi_lat_lon, 
                                        roi_window=self.roi_window, atcor=self.atcor)
            band_data, metadata, filename = reader.read_bands()
            for res, bands in band_data.items():
                for band, data in bands.items():
                    all_data[res][band].append((data, metadata[res]['transform'], metadata[res]['crs']))
                    if src_crs is None:
                        src_crs = metadata[res]['crs']
                if res in metadata:
                    bounds = array_bounds(
                        metadata[res]['height'], metadata[res]['width'], metadata[res]['transform']
                    )
                    bounds_list.append(bounds)
                    resolutions[res] = metadata[res]['transform'][0]
        minxs, minys, maxxs, maxys = zip(*bounds_list)
        global_bounds = (min(minxs), min(minys), max(maxxs), max(maxys))
        final_composite = defaultdict(dict)
        composite_metadata = {}
        for res, bands in all_data.items():
            pixel_size = resolutions[res]
            dst_width = int((global_bounds[2] - global_bounds[0]) / pixel_size)
            dst_height = int((global_bounds[3] - global_bounds[1]) / pixel_size)
            dst_transform = from_origin(global_bounds[0], global_bounds[3], pixel_size, pixel_size)
            for band, datasets in bands.items():
                common_canvas = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
                for data, transform, src_crs_band in datasets:
                    tmp_canvas = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
                    reproject(
                        source=data,
                        destination=tmp_canvas,
                        src_transform=transform,
                        src_crs=src_crs_band,
                        dst_transform=dst_transform,
                        dst_crs=src_crs,
                        resampling=Resampling.nearest,
                        src_nodata=None,
                        dst_nodata=np.nan,
                        init_dest_nodata=True
                    )
                    mask = ~np.isnan(tmp_canvas)
                    common_canvas[mask] = np.where(np.isnan(common_canvas[mask]), tmp_canvas[mask],
                                                   np.maximum(common_canvas[mask], tmp_canvas[mask]))
                final_composite[res][band] = common_canvas.astype(np.float32)
            composite_metadata[res] = {
                'dtype': np.float32,
                'count': len(bands),
                'crs': src_crs,
                'transform': dst_transform,
                'height': dst_height,
                'width': dst_width,
                'bands': list(bands.keys()),
                'start_date': date_codes[0],
                'end_date': date_codes[-1]
            }
        first_date = date_codes[0]
        last_date = date_codes[-1]
        prefix = f"Spatial_composite_{'_'.join(sorted(grid_codes))}_{first_date}_{last_date}"
        return final_composite, composite_metadata, prefix