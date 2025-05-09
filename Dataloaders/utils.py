import numpy as np
import torch
import matplotlib.pyplot as plt

def decode_seg_map_sequence(label_masks, dataset='rrhtdata'):
    """
    Decode a sequence of segmentation label masks into RGB masks and
    return as a torch tensor of shape (B, 3, H, W) with uint8 values [0–255].
    """
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    # shape (B, H, W, 3) → (B, 3, H, W)
    arr = np.stack(rgb_masks, axis=0).transpose((0, 3, 1, 2))
    return torch.from_numpy(arr)

def decode_segmap(label_mask, dataset='rrhtdata', plot=False):
    """
    Decode a single segmentation label mask (H×W) into an RGB image (H×W×3).
    Outputs uint8 values in [0,255].

    Args:
        label_mask (np.ndarray): 2D array of shape (H, W), integer class labels
        dataset (str): dataset name, only 'rrhtdata' supported
        plot (bool): if True, show the image with matplotlib

    Returns:
        np.ndarray: uint8 RGB image of shape (H, W, 3)
    """
    if dataset == 'rrhtdata':
        label_colours = get_rrhtdata_labels()
    else:
        raise NotImplementedError(f"Dataset '{dataset}' not supported")

    h, w = label_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, colour in enumerate(label_colours):
        mask = (label_mask == class_idx)
        rgb[mask, :] = colour

    if plot:
        plt.imshow(rgb)
        plt.axis('off')
        plt.show()
    return rgb

def encode_segmap(mask):
    """
    Encode an RGB segmentation mask into class indices.

    Args:
        mask (np.ndarray): H×W×3 uint8 array of colours

    Returns:
        np.ndarray: H×W array of class indices
    """
    mask = mask.astype(np.int32)
    label_mask = np.zeros(mask.shape[:2], dtype=np.int64)
    for idx, colour in enumerate(get_rrhtdata_labels()):
        matches = np.all(mask == colour, axis=-1)
        label_mask[matches] = idx
    return label_mask

def get_rrhtdata_labels():
    """
    Returns the mapping from class index to RGB colour for rrhtdata:
      class 0 → [  0,   0,   0]
      class 1 → [128,   0,   0]
    """
    return np.array([
        [  0,   0,   0],
        [128,   0,   0],
    ], dtype=np.uint8)
