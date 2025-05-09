import os
import rasterio
from rasterio.windows import Window

def split_tif_to_patches(
    filename: str,
    out_dir: str,
    patch_size: int = 512,
    overlap: int = 256
):
    """
    按照原始 gdal 代码的逻辑：只生成整齐网格上的全尺寸补丁，
    步长 = patch_size - overlap，编号到 (n_rows-1)_(n_cols-1)。
    """
    # 计算滑动步长
    step = patch_size - overlap

    # 创建输出目录
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(filename) as src:
        width, height = src.width, src.height

        # 原始代码等价：cols = width//step - 1，rows = height//step - 1
        n_cols = width // step - 1
        n_rows = height // step - 1

        for i in range(n_rows):         # i: 行编号，0…n_rows-1（原代码里的 i）
            for j in range(n_cols):     # j: 列编号，0…n_cols-1（原代码里的 j）
                # col_off = j*step, row_off = i*step
                window = Window(j * step, i * step, patch_size, patch_size)
                transform = src.window_transform(window)
                patch = src.read(window=window)

                # 复制并更新 profile，保持投影与地理变换正确
                profile = src.profile.copy()
                profile.update({
                    "height": patch_size,
                    "width": patch_size,
                    "transform": transform
                })

                out_path = os.path.join(out_dir, f"{i}_{j}.tif")
                with rasterio.open(out_path, 'w', **profile) as dst:
                    dst.write(patch)

if __name__ == "__main__":
    filename = r'F:\JGS_DATA\GF7_repro_4937_sub2423.tif'
    out_dir  = r'F:\JGS_DATA\IMG_4937_TIF'
    overlap = 256
    patch_size = 512
    split_tif_to_patches(filename, out_dir, patch_size, overlap)
    print("切片完成，总共生成：",
          ( (rasterio.open(filename).height // (patch_size - overlap) - 1) *
            (rasterio.open(filename).width  // (patch_size - overlap) - 1) )
    )
