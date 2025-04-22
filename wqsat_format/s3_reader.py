import os
import warnings
import numpy as np
import rioxarray
import xarray as xr
import rasterio
from rasterio.control import GroundControlPoint
from scipy.interpolate import RectBivariateSpline

from wqsat_format import atcor

class S3Reader():
    """
    Class to read data from S3
    """

    def __init__(self, tile_path, bands=None, roi_lat_lon=None, roi_window=None, atcor=True, output_format="GeoTIFF"):
        """
        Initialize the class with the bucket and key
        """

        warnings.filterwarnings("ignore", category=UserWarning, module="rioxarray._io")

        self.tile_path = tile_path.rstrip('/') + '/'  # Evitar doble barra
        self.atcor = atcor
        self.roi_lat_lon = roi_lat_lon
        self.roi_window = roi_window
        self.output_format = output_format

        if bands is None:
            bands = ['Oa01', 'Oa02', 'Oa03', 'Oa04', 'Oa05', 'Oa06', 'Oa07', 'Oa08', 'Oa09', 'Oa10', 
                     'Oa11', 'Oa12', 'Oa13', 'Oa14', 'Oa15', 'Oa16', 'Oa17', 'Oa18', 'Oa19', 'Oa20', 'Oa21']
        self.bands = bands

    def get_tr(self):
        """
        Read the latitude and longitude from S3
        """

        lat = rioxarray.open_rasterio(f'netcdf:{self.tile_path}geo_coordinates.nc:latitude')
        lon = rioxarray.open_rasterio(f'netcdf:{self.tile_path}geo_coordinates.nc:longitude')

        lat_scale = getattr(lat, 'scale_factor', 1)
        lon_scale = getattr(lon, 'scale_factor', 1)

        lon_corrected = lon.data[0] * lon_scale
        lat_corrected = lat.data[0] * lat_scale

        tr = rasterio.transform.from_bounds(
            west=np.min(lon_corrected), 
            south=np.min(lat_corrected),
            east=np.max(lon_corrected), 
            north=np.max(lat_corrected),
            width=lat.x.size, 
            height=lat.y.size)

        return lat_corrected, lon_corrected, tr

    def get_SunAngles(self, width, height):
        """
        Read the Sun angles from S3
        """

        file_path = os.path.join(self.tile_path, 'tie_geometries.nc')
        ds = xr.open_dataset(file_path)

        sza = ds["SZA"].values
        saa = ds["SAA"].values

        tie_x = np.linspace(0, width, sza.shape[1])
        tie_y = np.linspace(0, height, sza.shape[0])

        full_x = np.linspace(0, width, width)
        full_y = np.linspace(0, height, height)

        sza_interp = RectBivariateSpline(tie_y, tie_x, sza)(full_y, full_x)
        saa_interp = RectBivariateSpline(tie_y, tie_x, saa)(full_y, full_x)

        return sza_interp, saa_interp

    def read_bands(self):
        """
        Read the data from S3
        """

        valid_bands = [
            file for file in os.listdir(self.tile_path)
            if file.endswith('radiance.nc') and any(file.startswith(band) for band in self.bands)
        ]

        if not valid_bands:
            raise ValueError("No valid bands found in the Sentinel-3 tile.")

        valid_bands.sort(key=lambda x: self.bands.index(next(band for band in self.bands if x.startswith(band))))

        lat, lon, tr = self.get_tr()
        ds_list = []

        for b in valid_bands:
            banda = b.split('.')[0]
            print(f"Reading band {banda}...")
            A = rioxarray.open_rasterio(f'netcdf:{self.tile_path}{banda}.nc:{banda}')
            arr = A.data[0]

            if self.atcor:
                scale_factor = getattr(A, 'scale_factor', 1)
                sza, saa = self.get_SunAngles(A.sizes['x'], A.sizes['y'])
                arr = atcor.Atcor(arr, sza, saa, scale_factor).apply_all_corrections()

            ds_list.append(arr)

        arr_bands = np.array(ds_list)
        A = xr.DataArray(arr_bands, coords=[np.array(range(1, len(valid_bands)+1)), A.y.data, A.x.data], dims=A.dims)
        A.rio.write_crs("EPSG:4326", inplace=True)
        A.rio.write_transform(transform=tr, inplace=True)

        nof_gcp_x = np.arange(0, A.x.size, 25)
        nof_gcp_y = np.arange(0, A.y.size, 25)
        gcps = []
        for x in nof_gcp_x:
            for y in nof_gcp_y:        
                gcps.append(GroundControlPoint(row=int(y), col=int(x), x=float(lon[y, x]), y=float(lat[y, x])))

        tr_gcp = rasterio.transform.from_gcps(gcps)
        A.rio.write_transform(tr_gcp, inplace=True)
        # A = A.rio.reproject(dst_crs="EPSG:4326")

        meta = {
                "driver": "GTiff",
                "dtype": str(A.dtype),
                "nodata": A.rio.nodata,
                "width": A.rio.width,
                "height": A.rio.height,
                "count": A.sizes.get("band", 1),
                "crs": A.rio.crs,
                "transform": A.rio.transform()
            }

        if self.roi_lat_lon:
            A = A.rio.clip_box(
                minx=self.roi_lat_lon['W'], miny=self.roi_lat_lon['S'],
                maxx=self.roi_lat_lon['E'], maxy=self.roi_lat_lon['N']
            )

        elif self.roi_window:
            col_off = self.roi_window['xmin']
            row_off = self.roi_window['ymin']
            width = (self.roi_window['xmax'] - self.roi_window['xmin'])
            height = (self.roi_window['ymax'] - self.roi_window['ymin'])

            A = A.isel(x=slice(col_off, col_off + width),
                       y=slice(row_off, row_off + height))

        # Exportar GeoTIFF
        file = os.path.basename(os.path.normpath(self.tile_path))
        file = file.replace(".SEN3", ".tif")
        output_file = os.path.join(self.tile_path, file)

        A.rio.to_raster(output_file, recalc_transform=True)
        print(f'Tif of file {self.tile_path} saved\n')