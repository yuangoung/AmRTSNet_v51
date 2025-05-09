import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from Dataloaders.utils import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, directory):
        """
        TensorBoard

        Args:
            directory (str): TensorBoard
        """
        self.directory = directory

    def create_summary(self):
        """
        SummaryWriter
        Returns:
            SummaryWriter: TensorBoard
        """
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        """
        TensorBoard。
        Args:
            writer (SummaryWriter): TensorBoard
            dataset (str): decode_seg_map_sequence
            image (Tensor):  (B, C, H, W)
            target GT (Tensor): (B, 1, H, W)
            output (Tensor):  logits (B, num_classes, H, W)
            global_step (int):  TensorBoard x
        """
        # 1. normalize=True
        grid_image = make_grid(
            image[:3].clone().cpu().data,
            nrow=3,
            normalize=True
        )
        writer.add_image('Image', grid_image, global_step)

        # 2. pred_indices —— decode label，value_range=(0,255)
        pred_indices = torch.max(output[:3], dim=1)[1]   # (3, H, W)
        pred_maps = decode_seg_map_sequence(
            pred_indices.detach().cpu().numpy(),
            dataset=dataset
        )
        grid_pred = make_grid(
            pred_maps,
            nrow=3,
            normalize=False,
            value_range=(0, 255)
        )
        writer.add_image('Predicted label', grid_pred, global_step)

        # 3. GT ——  value_range=(0,255)
        gt_indices = torch.squeeze(target[:3], dim=1)   # (3, H, W)
        gt_maps = decode_seg_map_sequence(
            gt_indices.detach().cpu().numpy(),
            dataset=dataset
        )
        grid_gt = make_grid(
            gt_maps,
            nrow=3,
            normalize=False,
            value_range=(0, 255)
        )
        writer.add_image('Groundtruth label', grid_gt, global_step)
