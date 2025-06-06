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
        A = xr.DataArray(
            arr_bands,
            coords=[np.arange(len(valid_bands)), np.arange(lat.shape[0]), np.arange(lat.shape[1])],
            dims=("band", "y", "x"))
        A.rio.write_crs("EPSG:4326", inplace=True)

        # Escribimos GCPs
        gcps = []
        for x in range(0, lat.shape[1], 25):
            for y in range(0, lat.shape[0], 25):
                gcps.append(GroundControlPoint(row=y, col=x, x=lon[y, x], y=lat[y, x], z=0.0))

        A.rio.write_crs("EPSG:4326", inplace=True)
        A.rio.write_gcps(gcps, "EPSG:4326", inplace=True)
        A = A.rio.reproject("EPSG:4326")

        # A = xr.DataArray(arr_bands, coords=[np.array(range(1, len(valid_bands)+1)), A.y.data, A.x.data], dims=A.dims)
        # # A.rio.write_crs("EPSG:4326", inplace=True)
        # # A.rio.write_transform(transform=tr, inplace=True)

        # nof_gcp_x = np.arange(0, A.x.size, 25)
        # nof_gcp_y = np.arange(0, A.y.size, 25)
        # gcps = []
        # for x in nof_gcp_x:
        #     for y in nof_gcp_y:        
        #         gcps.append(GroundControlPoint(row=int(y), col=int(x), x=float(lon[y, x]), y=float(lat[y, x]), z=0.0))

        # tr_gcp = rasterio.transform.from_gcps(gcps)
        # A.rio.write_crs("EPSG:4326", inplace=True)
        # A.rio.write_transform(tr_gcp, inplace=True)

        # # Finalmente, reproyectamos (esto interpolará a una grilla regular en EPSG:4326)
        # A = A.rio.reproject(dst_crs="EPSG:4326")
        # A.rio.write_transform(tr_gcp)
        # A = A.rio.reproject(dst_crs="EPSG:4326")
        # A = A.rio.reproject(dst_crs="EPSG:4326", gcps=gcps, **{"SRC_METHOD": "GCP_TPS"})

        print("📌 Imagen reproyectada:")
        print(f" - Lon min: {A.x.min().item():.6f}")
        print(f" - Lon max: {A.x.max().item():.6f}")
        print(f" - Lat min: {A.y.min().item():.6f}")
        print(f" - Lat max: {A.y.max().item():.6f}")

        print("📌 ROI solicitado:")
        print(f" - Lon W (minx): {self.roi_lat_lon['W']}")
        print(f" - Lon E (maxx): {self.roi_lat_lon['E']}")
        print(f" - Lat S (miny): {self.roi_lat_lon['S']}")
        print(f" - Lat N (maxy): {self.roi_lat_lon['N']}")

        # Extra: comprueba si hay intersección
        roi = self.roi_lat_lon
        intersecta = not (
            roi["E"] < A.x.min() or roi["W"] > A.x.max() or
            roi["N"] < A.y.min() or roi["S"] > A.y.max()
        )
        print(f"❓¿ROI intersecta imagen?: {'✅ Sí' if intersecta else '❌ No'}")

        # Recorte de bandas si aplica ROI
        if self.roi_lat_lon:
            A_clip = A.rio.clip_box(
                minx=self.roi_lat_lon['W'], miny=self.roi_lat_lon['S'],
                maxx=self.roi_lat_lon['E'], maxy=self.roi_lat_lon['N']
            )
            y_idx = np.where((A.y >= A_clip.y.min()) & (A.y <= A_clip.y.max()))[0]
            x_idx = np.where((A.x >= A_clip.x.min()) & (A.x <= A_clip.x.max()))[0]
            arr_bands = arr_bands[:, y_idx[0]:y_idx[-1]+1, x_idx[0]:x_idx[-1]+1]
            A = A_clip

        elif self.roi_window:
            col_off = self.roi_window['xmin']
            row_off = self.roi_window['ymin']
            width = self.roi_window['xmax'] - self.roi_window['xmin']
            height = self.roi_window['ymax'] - self.roi_window['ymin']
            arr_bands = arr_bands[:, row_off:row_off + height, col_off:col_off + width]
            A = A.isel(x=slice(col_off, col_off + width), y=slice(row_off, row_off + height))

        meta = {
            "bands": valid_bands,
            "height": arr_bands.shape[1],
            "width": arr_bands.shape[2],
            "transform": A.rio.transform(),
            "crs": A.rio.crs
        }

        filename = os.path.basename(os.path.normpath(self.tile_path)).split('.')[0]
        return arr_bands, meta, filename