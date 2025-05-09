import os
import argparse
import time
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader

class TiffDataset(Dataset):
    """four bands tif images"""
    def __init__(self, folder):
        self.paths = [os.path.join(folder, f)
                      for f in os.listdir(folder)
                      if f.lower().endswith('.tif')]
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        with rasterio.open(self.paths[idx]) as src:
            arr = src.read().astype('float32')  # (C, H, W)
        return torch.from_numpy(arr)          # Tensor shape: (C, H, W)


def update_stats(mean, M2, count, batch):
    """
    Welford 在线更新算法：
      mean: 当前累计均值，shape [C]
      M2: 当前累计二阶中心动差，shape [C]
      count: 当前累计像素数
      batch: [B, C, HW]
    返回：更新后的 (mean, M2, count)
    """
    B, C, HW = batch.shape
    for b in range(B):
        data = batch[b]               # [C, HW]
        n_b = HW
        m_b = data.mean(dim=1)       # [C]
        var_b = data.var(dim=1, unbiased=False)  # [C]
        M2_b = var_b * n_b

        n_a = count
        m_a = mean.clone()
        # 更新总样本数
        count = n_a + n_b
        # 差值
        delta = m_b - m_a
        # 更新均值
        mean = m_a + delta * (n_b / count)
        # 更新 M2
        M2 = M2 + M2_b + delta * delta * (n_a * n_b / count)
    return mean, M2, count


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-band mean and std over all TIFFs with PyTorch+CUDA")
    parser.add_argument('--folder', type=str,
                        default=r'F:\JGS_DATA\IMG_4937_TIF',
                        help="Path to folder containing .tif images")
    parser.add_argument('--batch-size', type=int, default=24,
                        help="Number of images per batch")
    parser.add_argument('--workers', type=int, default=4,
                        help="Number of DataLoader workers")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    dataset = TiffDataset(args.folder)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=args.workers,
                         pin_memory=True)

    # calculate
    C = 4  # 四波段
    mean  = torch.zeros(C, device=device)
    M2    = torch.zeros(C, device=device)
    count = 0

    start = time.time()
    for imgs in loader:
        # imgs: [B, C, H, W]
        B, C, H, W = imgs.shape
        imgs = imgs.to(device).view(B, C, -1)  # [B, C, HW]
        mean, M2, count = update_stats(mean, M2, count, imgs)

    #  std
    var = M2 / count
    std = torch.sqrt(var)

    elapsed = time.time() - start

    # print
    for i, (m, s) in enumerate(zip(mean.tolist(), std.tolist())):
        print(f"Band {i+1}: mean = {m:.6f}, std = {s:.6f}")
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"Elapsed time: {int(hrs)}h {int(mins)}m {int(secs)}s")

if __name__ == "__main__":
    main()