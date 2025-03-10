import os
import warnings
import numpy as np
import rioxarray
import xarray as xr
import rasterio
from rasterio.control import GroundControlPoint
from scipy.interpolate import RectBivariateSpline

from wqsat_format import s3_atcor

class S3Reader():
    """
    Class to read data from S3
    """

    def __init__(self, tile_path, bands=None, roi_lat_lon=None, roi_window=None, atcor=True):
        """
        Initialize the class with the bucket and key
        """
        self.tile_path = tile_path.rstrip('/') + '/'  # Evitar doble barra
        self.atcor = atcor
        self.roi_lat_lon = roi_lat_lon
        self.roi_window = roi_window

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

        # Obtener scale_factor si existe
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
    
    def get_window_crop(self):

        xmin, xmax, ymin, ymax = self.roi_window
        lat, lon, tr = self.get_tr()

        lat_crop = lat[xmin:xmax, ymin:ymax]
        lon_crop = lon[xmin:xmax, ymin:ymax]

        tr = rasterio.transform.from_bounds(
            west=np.min(lon_crop), 
            south=np.min(lat_crop),
            east=np.max(lon_crop), 
            north=np.max(lat_crop),
            width=lat.x.size, 
            height=lat.y.size)
        
        return lat_crop, lon_crop, tr
    
    def get_SunAngles(self, width, height):
        """
        Read the Sun angles from S3
        """

        file_path = os.path.join(self.tile_path, 'tie_geometries.nc')
        ds = xr.open_dataset(file_path)

        # Extraer SZA y SAA
        sza = ds["SZA"].values
        saa = ds["SAA"].values

        # Crear mallas de coordenadas originales
        tie_x = np.linspace(0, width, sza.shape[1])  # Puntos de atadura en X
        tie_y = np.linspace(0, height, sza.shape[0])  # Puntos de atadura en Y

        # Crear mallas de destino a resolución completa
        full_x = np.linspace(0, width, width)  # Resolución completa en X
        full_y = np.linspace(0, height, height)  # Resolución completa en Y

        # Interpolación bilineal
        sza_interp = RectBivariateSpline(tie_y, tie_x, sza)(full_y, full_x)
        saa_interp = RectBivariateSpline(tie_y, tie_x, saa)(full_y, full_x)

        return sza_interp, saa_interp

    def read_bands(self):
        """
        Read the data from S3
        """

        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
        ds_list= []

        if self.roi_lat_lon:
            lat, lon, tr = self.get_window_crop()
        return lat, lon, tr

        # for b in bands:

        #     # Read the data
        #     band = f'{b}_radiance'
        #     ds = rioxarray.open_rasterio(f'netcdf:{self.tile_path}{band}.nc:{band}')
        #     arr = ds.values.astype(np.uint16)

        #     if self.atcor:

        #         # Apply ATCOR
        #         scale_factor = getattr(ds, 'scale_factor', 1)
        #         sza, saa = self.get_SunAngles(ds.sizes['x'], ds.sizes['y'])
                
        #         # Create AtcorSentinel3 object
        #         arr = s3_atcor.AtcorSentinel3(arr, sza, saa, scale_factor).apply_all_corrections()

        #     # Append the data to the list
        #     ds_list.append(arr)

        # arr_bands = np.squeeze(np.array(ds_list))

        # ds = xr.DataArray(arr_bands, 
        #                   coords = [np.array(range(1,22)), ds.y.data, ds.x.data], 
        #                   dims = ('band', 'y', 'x'))
    
        # lat, lon, tr = self.get_tr()
        # ds.rio.write_crs("EPSG:4326", inplace=True)
        # ds.rio.write_transform(transform=tr, inplace=True)

        # nof_gcp_x = np.arange(0, ds.x.size, 100)
        # nof_gcp_y = np.arange(0, ds.y.size, 100)
        # gcps = []
        # id = 0

        # for x in nof_gcp_x:
        #     for y in nof_gcp_y:        
        #         gcps.append(GroundControlPoint(
        #             row=y, col=x, 
        #             x=lon[y, x],
        #             y=lat[y, x],
        #             id=id))
        #         id += 1

        # tr_gcp = rasterio.transform.from_gcps(gcps)
        # ds = ds.rio.reproject(dst_crs="EPSG:4326", transform=tr_gcp, gcps=gcps, **{"SRC_METHOD": "GCP_TPS"})

        # file = self.tile.split('.')[0] + '.tif'
        # output_file = os.path.join(self.tile_path, file)
        # ds.rio.to_raster(output_file, recalc_transform=False)
        # print('Tif of file {} saved'.format(self.tile_path))
