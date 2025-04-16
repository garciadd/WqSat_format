# WqSat_format
--------------

## Overview

**WqSat_format** is a Python package developed for preprocessing Sentinel-2 and Sentinel-3 satellite images, with a focus on inland water quality monitoring. It is part of the WqSat project and provides tools to:

- Crop satellite images based on geographic coordinates.
- Generate mosaics from multiple images.
- Export processed images in NetCDF or GeoTIFF format.
- Apply basic atmospheric corrections such as sun angle and dark object subtraction (DOS) to Level-1 images.
- Perform essential satellite image processing tasks.

🚧 This repository is under active development. Features and functionalities may change as improvements are made. If you encounter any issues or have suggestions, feel free to contribute or report them via Issues.

---

## Authors
Daniel García-Díaz & Fernando Aguilar
Email: (garciad & aguilarf) @ifca.unican.es
Spanish National Research Council (CSIC); Institute of Physics of Cantabria (IFCA)
Advanced Computing and e-Science

---

## Installation
To install **WqSat_format**, clone the repository and use `pip` to install the package:

```bash
# Clone the repository
git clone https://github.com/garciadd/WqSat_format.git

# Navigate to the package directory
cd WqSat_format

# Install the package
pip install .
```

---

## Requirements
Make sure the following Python libraries are installed:

- numpy
- xarray
- rioxarray
- rasterio
- scipy

You can install all dependencies using:

```bash
# install all dependencies
pip install -r requirements.txt
```

---

## Usage
### Crop Image by Geographic Coordinates
```python
from wqsat_format import S3Reader

# Define region of interest in lat/lon
roi_lat_lon = {
    "N": 39.69779,
    "S": 38.74476,
    "E": 0.770585,
    "W": -0.44578
}

# Path to the Sentinel-3 image folder
tile_path = "/path/to/image.SEN3/"

# Create an instance of the reader
reader = S3Reader(tile_path=tile_path, roi_lat_lon=roi_lat_lon)

# Read and process the bands
reader.read_bands()
```

### Crop Image by Geographic Coordinates
```python
from wqsat_format import S3Reader

# Define pixel window
roi_window = {
    "xmin": 100,
    "ymin": 200,
    "xmax": 500,
    "ymax": 600
}

# Path to the Sentinel-3 image folder
tile_path = "/path/to/image.SEN3/"

# Create an instance of the reader
reader = S3Reader(tile_path=tile_path, roi_window=roi_window)

# Read and process the bands
reader.read_bands()
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
