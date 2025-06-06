import os
import numpy as np
import xml.etree.ElementTree as ET

class Atcor:
    def __init__(self, arr_band, sun_zenith, sun_azimuth, scale_factor=0.00001):
        """
        Atmospheric Correction Class for Sentinel-2 L1C images.
        Atmospheric Correction Class for Sentinel-3 OLCI L1 products.
        
        Parameters:
        - arr (np.ndarray): 2D array with shape (height, width), representing a single band.
        - sun_zenith (float): Solar zenith angle in degrees.
        - sun_azimuth (float): Solar azimuth angle in degrees.
        - scale_factor (int, optional): Scaling factor for reflectance values (default: 0.00001).
        """
        
        self.arr = np.where((arr_band == 65535), np.nan, arr_band)
        self.sza = np.radians(sun_zenith)  # Convert SZA to radians
        self.saa = np.radians(sun_azimuth)  # Convert SAA to radians
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