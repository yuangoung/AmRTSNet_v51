import os
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

class Tiff:
    def __init__(self, filename=None):
        self.filename = filename
        self.im_band = None
        self.im_width = None
        self.im_height = None
        self.im_proj = None
        self.im_geotrans = None
        self.im_data = None
        self.overall_img = None

    def read_img(self):
        if not self.filename:
            raise ValueError("No filename provided for reading image.")

        dataset = gdal.Open(self.filename)
        if dataset is None:
            raise FileNotFoundError(f"Cannot open {self.filename}")

        self.im_width = dataset.RasterXSize
        self.im_height = dataset.RasterYSize
        self.im_geotrans = dataset.GetGeoTransform()
        self.im_proj = dataset.GetProjection()
        data = dataset.ReadAsArray(0, 0, self.im_width, self.im_height)
        # Select first three bands if multiband
        if data.ndim == 3:
            self.im_data = data[[0, 1, 2], :, :]
            self.im_band = self.im_data.shape[0]
        else:
            self.im_data = data
            self.im_band = 1
        del dataset

    def write_img(self, out_tif_name, n_band=1):
        driver = gdal.GetDriverByName("GTiff")

        if n_band == 3:
            data = self.im_data
            out_tif = driver.Create(
                out_tif_name + '.tif',
                data.shape[2], data.shape[1], data.shape[0],
                gdal.GDT_Float32
            )
            out_tif.SetProjection(self.im_proj)
            out_tif.SetGeoTransform(self.im_geotrans)
            for i in range(data.shape[0]):
                out_tif.GetRasterBand(i+1).WriteArray(data[i])
            del out_tif
        else:
            data = self.overall_img
            out_tif = driver.Create(
                out_tif_name + '.tif',
                data.shape[1], data.shape[0], 1,
                gdal.GDT_Float32
            )
            out_tif.SetProjection(self.im_proj)
            out_tif.SetGeoTransform(self.im_geotrans)
            out_tif.GetRasterBand(1).WriteArray(data)
            del out_tif

    def mosaic(self, folder_path, patch_size=512, overlap=256, init_shape=None, nodata_value=0):
        # Determine overall size if not set
        if init_shape:
            rows, cols = init_shape
        else:
            # guess from patches: find max i,j
            max_i = max_j = 0
            for fname in os.listdir(folder_path):
                i, j = map(int, fname.split('_')[:2])
                max_i = max(max_i, i)
                max_j = max(max_j, j)
            rows = (max_i + 1) * (patch_size - overlap) + overlap
            cols = (max_j + 1) * (patch_size - overlap) + overlap

        self.overall_img = np.full((rows, cols), nodata_value, dtype=np.float32)

        for file in sorted(os.listdir(folder_path)):
            if not file.lower().endswith(('.png', '.jpg', '.tif')):
                continue
            i, j = map(int, file.split('_')[:2])
            path = os.path.join(folder_path, file)
            sub_img = plt.imread(path)
            # convert to single channel mask
            if sub_img.ndim == 3:
                sub_img = np.mean(sub_img, axis=2)
            sub_img = np.where(sub_img != nodata_value, 1.0, 0.0)

            i0 = i * (patch_size - overlap)
            j0 = j * (patch_size - overlap)
            h, w = sub_img.shape
            self.overall_img[i0:i0+h, j0:j0+w] = sub_img

        # Display mosaic
        plt.imshow(self.overall_img, cmap='gray')
        plt.title('Mosaic Result')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # Original GF7 image (for georeferencing)
    filename = r'F:\JGS_DATA\GF7_repro_4937_sub2423.tif'
    output_folder = r'F:\JGS_DATA\output_masks'
    output_tif = r'F:\JGS_DATA\mosaic_result'

    img = Tiff(filename)
    img.read_img()
    # Build mosaic from masks
    img.mosaic(folder_path=output_folder, patch_size=512, overlap=256)
    # Save as GeoTIFF
    img.write_img(out_tif_name=output_tif, n_band=1)