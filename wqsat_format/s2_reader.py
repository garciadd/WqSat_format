import os, re
import glob
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
from collections import defaultdict
from typing import List
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.transform import array_bounds, from_origin
import rasterio.merge
from rasterio.warp import reproject, Resampling, calculate_default_transform, transform_bounds
import xarray as xr

from wqsat_format import atcor

class S2Reader:
    def __init__(self, tile_path, bands=None, roi_lat_lon=None, roi_window=None, atcor=True, 
                 temporal_composite=None, crs="EPSG:4326", output_format="GeoTIFF"):

        self.tile_path = tile_path
        self.roi_lat_lon = roi_lat_lon
        self.roi_window = roi_window
        self.atcor = atcor
        self.temporal_composite = temporal_composite
        self.crs = crs
        self.output_format = output_format
        
        # Default to all available bands if none are specified
        self.bands = bands or ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 
                               'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        
    def calculate_window(self, src, resolution):

        factor = int(resolution) // 10 # Adjust window size based on resolution factor

        if self.roi_lat_lon:
            # Convert geographic coordinates to image bounds
            W, N, E, S = self.roi_lat_lon['W'], self.roi_lat_lon['N'], self.roi_lat_lon['E'], self.roi_lat_lon['S']
            bounds = transform_bounds(self.crs, src.crs, float(W), float(S), float(E), float(N))

            # Ajustar los bounds a los límites de la imagen
            left, bottom, right, top = src.bounds
            new_bounds = (
                max(bounds[0], left),
                max(bounds[1], bottom),
                min(bounds[2], right),
                min(bounds[3], top)
            )

            # Crear window a partir de bounds ajustados
            window = from_bounds(*new_bounds, transform=src.transform)

            return window, new_bounds

        elif self.roi_window:

            # Use the provided pixel window
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
            sza =  float(sun_angle_node.find("ZENITH_ANGLE").text),
            saa =  float(sun_angle_node.find("AZIMUTH_ANGLE").text)
    
        return sza, saa

    def read_bands(self):

        try:
            # Locate the IMG_DATA folder within the Sentinel-2 tile structure
            granule_path = glob.glob(os.path.join(self.tile_path, "GRANULE", "*", "IMG_DATA"))[0]
            jp2_files = {os.path.basename(f).split("_")[-1].split(".")[0]: f for f in glob.glob(os.path.join(granule_path, "*.jp2"))}
        except IndexError:
            raise ValueError("Granule path not found in the Sentinel-2 tile directory.")
        
        valid_bands = [b for b in self.bands if b in jp2_files]
        valid_bands.sort(key=lambda x: self.bands.index(next(b for b in self.bands if x.startswith(b))))
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
                        window, new_bounds = self.calculate_window(src, int(res))
                        data = src.read(1, window=window)

                        if self.atcor:
                            # Apply ATCOR corrections
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
                                "bounds": new_bounds if new_bounds else src.bounds
                            })
                            metadata[res] = meta
                except Exception as e:
                    raise RuntimeError(f"Error reading band {band} at {res}m resolution: {e}")    
        return arr_bands, metadata

    def compose_temporal_tiles(self, tile_paths: list[str]):
        if self.temporal_composite not in ["median", "max"]:
            raise ValueError("Invalid composite type. Use 'median' or 'max'.")

        grid_codes = {re.search(r"T\d{2}[A-Z]{3}", os.path.basename(path)).group() for path in tile_paths}
        if len(grid_codes) != 1:
            raise ValueError(f"Tiles do not share the same grid: {grid_codes}")

        tile_data = []
        for path in tile_paths:
            print(f"Reading tile {os.path.basename(os.path.normpath(path))}...")
            date_str = re.search(r"\d{8}T\d{6}", path).group()[:8]
            date = datetime.strptime(date_str, "%Y%m%d")
            self.tile_path = path
            band_data, metadata = self.read_bands()
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
        self.tile_path = os.path.commonpath(tile_paths)
        self.export_data(final_composite, composite_metadata, prefix)
    
    def compose_spatial_tiles(self, tile_paths: list[str]):
        if len(tile_paths) < 2:
            raise ValueError("At least two tile paths are required for spatial composition.")
        grid_codes = {re.search(r"T\d{2}[A-Z]{3}", os.path.basename(path)).group() for path in tile_paths}
        date_codes = sorted([re.search(r"\d{8}T\d{6}", os.path.basename(path)).group()[:8] for path in tile_paths])
        all_data = defaultdict(lambda: defaultdict(list))
        bounds_list = []
        src_crs = None
        resolutions = {}
        for path in tile_paths:
            self.tile_path = path
            print(f"Reading tile {os.path.basename(os.path.normpath(self.tile_path))}...")
            bands_data, metadata = self.read_bands()
            for res, bands in bands_data.items():
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
        self.tile_path = os.path.commonpath(tile_paths)
        self.export_data(final_composite, composite_metadata, prefix)

    def export_data(self, arr_bands, metadata, prefix=None):
        file = prefix if prefix else os.path.basename(os.path.normpath(self.tile_path)).split('.')[0]
        for res, meta in metadata.items():
            output_file = os.path.join(self.tile_path, f"{file}_{res}m.{self.output_format.lower()}")
            meta.update({
                "count": len(meta["bands"]),
                "dtype": np.float32
            })
            if self.output_format.lower() == "geotiff":
                meta.update({"driver": "GTiff"})
                with rasterio.open(output_file, "w", **meta) as dst:
                    for i, band in enumerate(meta["bands"], start=1):
                        dst.write(arr_bands[res][band], i)
                        dst.set_band_description(i, band)
            elif self.output_format.lower() == "netcdf":
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
                        "description": f"Sentinel-2 data for {file} at {res}m resolution",
                        "date_range": f"{meta.get('start_date', 'unknown')} to {meta.get('end_date', 'unknown')}"
                    }
                )
                ds.rio.write_crs(meta["crs"], inplace=True)
                ds.to_netcdf(output_file)
            else:
                raise ValueError("Unsupported output format. Use 'GeoTIFF' or 'NetCDF'.")
