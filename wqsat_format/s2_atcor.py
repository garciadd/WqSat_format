import numpy as np

class Atcor:
    def __init__(self, arr_bands, sun_zenith, sun_azimuth, scale_factor=10000):
        """
        Atmospheric Correction Class for Sentinel-2 L1C images.
        
        Parameters:
        - arr_bands (dict): Dictionary containing image bands organized by resolution.
          Example structure:
          {
              '10': {'B4': arr_b4, 'B3': arr_b3, 'B2': arr_b2, ...},
              '20': {'B5': arr_b5, 'B8': arr_b8, ...},
              '60': {'B1': arr_b1, ...}
          }
        - sun_zenith (float): Solar zenith angle in degrees.
        - sun_azimuth (float): Solar azimuth angle in degrees.
        - scale_factor (int, optional): Scaling factor for reflectance values (default: 10000).
        """
        
        self.arr_bands = arr_bands  # Dictionary with resolutions as keys and band dictionaries as values
        self.sun_zenith = sun_zenith
        self.sun_azimuth = sun_azimuth
        self.scale_factor = scale_factor
    
    def apply_scale_factor(self):
        """Normalize the reflectance values by dividing by the scale factor."""
        for resolution, bands in self.arr_bands.items():
            for band, arr in bands.items():
                self.arr_bands[resolution][band] = arr / self.scale_factor
        return self.arr_bands
    
    def sun_angle_correction(self):
        """Apply solar angle correction using the cosine of the sun zenith angle."""
        sun_zenith_rad = np.radians(self.sun_zenith)
        for resolution, bands in self.arr_bands.items():
            for band, arr in bands.items():
                self.arr_bands[resolution][band] = arr / np.cos(sun_zenith_rad)
        return self.arr_bands
    
    def dark_object_subtraction(self):
        """Apply DOS (Dark Object Subtraction) correction by subtracting the minimum value in each band."""
        for resolution, bands in self.arr_bands.items():
            for band, arr in bands.items():
                dark_object_value = np.percentile(arr, 1)  # Approximate dark object
                arr = arr - dark_object_value
                self.arr_bands[resolution][band] = np.clip(arr, 0, None)  # Ensure no negatives
        return self.arr_bands
    
    def apply_all_corrections(self):
        """Apply all corrections in sequence."""
        self.apply_scale_factor()
        self.sun_angle_correction()
        self.dark_object_subtraction()
        return self.arr_bands