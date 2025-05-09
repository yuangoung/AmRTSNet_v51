import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import rasterio
from modeling.AmRTSNet import AmRTSNet

def get_rrhtdata_labels():
    """Load the mapping that associates rrhtdata classes with label colors
    Returns:
        np.ndarray with dimensions (2, 3)
    """
    return np.asarray([
        [0,   0,   0],   # class 0: background
        [128, 0,   0],   # class 1: RTSs
    ], dtype=np.uint8)

def main():
    parser = argparse.ArgumentParser(description="PyTorch AmRTSNet Inference")
    parser.add_argument('--in-path',   type=str, default=r'F:\JGS_DATA\IMG_4937_TIF')
    parser.add_argument('--out-path',  type=str, default=r'F:\JGS_DATA\output_masks')
    parser.add_argument('--ckpt',      type=str,
                        default=r'F:\AmRTSNet_v51\run\rrhtdata\AmRTSNet-mobilenet\model_best.pth.tar')
    parser.add_argument('--backbone',  type=str, default='mobilenet')
    parser.add_argument('--out-stride',type=int, default=8)
    parser.add_argument('--num-classes',type=int, default=2)
    parser.add_argument('--no-cuda',   action='store_true', default=False)
    parser.add_argument('--gpu-ids',   type=str, default='0')
    parser.add_argument('--dataset',   type=str, default='rrhtdata')
    args = parser.parse_args()

    # GPU
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
    device = torch.device(f'cuda:{args.gpu_ids[0]}' if args.cuda else 'cpu')

    # load_state_dict checkpoint
    model = AmRTSNet(num_classes=args.num_classes,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=False,
                    freeze_bn=False)
    checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    os.makedirs(args.out_path, exist_ok=True)

    # normalization ：(DN - mean) / std (patch calculate)）
    mean = np.array([1506, 2009, 2092, 3103], dtype=np.float32)
    std  = np.array([361,  447,  450,  587], dtype=np.float32)

    #  LOAD label colors
    label_colors = get_rrhtdata_labels()

    for filename in os.listdir(args.in_path):
        if not filename.lower().endswith('.tif'):
            continue

        input_path  = os.path.join(args.in_path, filename)
        output_path = os.path.join(
            args.out_path,
            os.path.splitext(filename)[0] + '_mask.png'
        )

        # FOUR BANDS TIFF IMG (4, H, W)
        with rasterio.open(input_path) as src:
            img = src.read().astype(np.float32)  # shape: (4, H, W)
        # TRANS (H, W, 4)
        img = np.transpose(img, (1, 2, 0))

        # normalization ：(DN - mean) / std
        img = (img - mean) / std

        # TRANS tensor (1, 4, H, W)
        tensor = torch.from_numpy(img.transpose(2, 0, 1)) \
                      .unsqueeze(0) \
                      .to(device=device, dtype=torch.float32)

        # PRE
        with torch.no_grad():
            out = model(tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = F.interpolate(out,
                                size=tensor.shape[2:],  # RECOVER (H, W)
                                mode='bilinear',
                                align_corners=True)
            # CLASS
            pred = out.argmax(dim=1).squeeze(0).cpu().numpy()  # shape: (H, W)

        # LABEL RGB
        color_mask = label_colors[pred]  # shape: (H, W, 3)
        img_pil = Image.fromarray(color_mask)
        img_pil.save(output_path)

        print(f"Processed {filename} → {os.path.basename(output_path)}")

if __name__ == "__main__":
    main()
