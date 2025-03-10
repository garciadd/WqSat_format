import numpy as np

class AtcorSentinel3:
    def __init__(self, arr, sza, saa, scale_factor):
        """
        Atmospheric Correction Class for Sentinel-3 OLCI L1 products.

        Parameters:
        - arr (np.ndarray): 3D array with shape (1, height, width), representing a single band.
        - sza (np.ndarray): 3D array with shape (1, height, width), Solar Zenith Angle in degrees.
        - saa (np.ndarray): 3D array with shape (1, height, width), Solar Azimuth Angle in degrees.
        - scale_factor (float): Scaling factor to convert DN to reflectance.
        """
        # Reemplazar valores no válidos (0 y 65535) por NaN
        self.arr = np.where((arr == 65535), np.nan, arr)
        self.sza = np.radians(sza)  # Convertir SZA a radianes
        self.saa = np.radians(saa)  # Convertir SAA a radianes
        self.scale_factor = scale_factor
    
    def apply_scale_factor(self):
        """Normalize the reflectance values by dividing by the scale factor."""
        self.arr *= self.scale_factor
        return self.arr
    
    def sun_angle_correction(self):
        """Apply solar angle correction using the solar zenith angle (SZA)."""
        if self.sza is None:
            raise ValueError("Solar zenith angle array is missing.")
        
        self.arr /= np.cos(self.sza)
        return self.arr
    
    def dark_object_subtraction(self):
        """Apply DOS (Dark Object Subtraction) correction by subtracting the minimum valid value."""
        valid_pixels = self.arr[~np.isnan(self.arr)]
        if valid_pixels.size > 0:
            dark_object_value = np.percentile(valid_pixels, 1)  # Percentil 1% para encontrar píxeles oscuros
            self.arr -= dark_object_value
        
        self.arr = np.clip(self.arr, 0, None)  # Asegurar que no haya valores negativos
        return self.arr
    
    def apply_all_corrections(self):
        """Apply all corrections (scale factor, solar angle, and DOS) in sequence."""
        self.apply_scale_factor()
        self.sun_angle_correction()
        self.dark_object_subtraction()
        return self.arr