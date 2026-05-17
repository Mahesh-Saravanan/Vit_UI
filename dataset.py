"""
dataset.py — RICO UI component detection dataset.

Preprocessing pipeline per image:
  1. Load original image (arbitrary resolution).
  2. Letterbox-pad to a square of side max(orig_w, orig_h) — content at top-left.
  3. Record orig_w, orig_h, max_dim before any resizing.
  4. Normalise bounding boxes in two steps:
         x_norm = (x / orig_w) * (orig_w / max_dim)  =  x / max_dim
         y_norm = (y / orig_h) * (orig_h / max_dim)  =  y / max_dim
     Clamped to [0, 1].  These coords now live in the padded-square space.
  5. Sort children by (y1_norm, x1_norm) — top-left → bottom-right reading order.
  6. Resize padded image to IMAGE_SIZE × IMAGE_SIZE and apply ImageNet normalisation.
     Normalised coords are already scale-invariant, so no further adjustment needed.

Dataset.__getitem__ returns:
    image    : FloatTensor [3, IMAGE_SIZE, IMAGE_SIZE]
    sequence : list of (x1, y1, x2, y2, class_idx) float/int tuples  (variable length)
    length   : int  — number of children in this sample

collate_fn pads sequences in a batch to the same length (PAD = 0) and returns a
boolean padding mask (True = PAD position).
"""

import os
import json
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import config

# ── Class mapping ─────────────────────────────────────────────────────────────

CLASS_TO_IDX: dict = {cls: i for i, cls in enumerate(config.UI_CLASSES)}
IDX_TO_CLASS: dict = {i: cls for cls, i in CLASS_TO_IDX.items()}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _letterbox_and_normalise(
    img: Image.Image,
    children: list,
) -> Tuple[torch.Tensor, List[Tuple]]:
    """
    Letterbox-pad img to a square, normalise bbox coords, sort children,
    resize to IMAGE_SIZE × IMAGE_SIZE, and return (image_tensor, sequence).
    """
    orig_w, orig_h = img.size                           # PIL gives (width, height)
    max_dim = max(orig_w, orig_h)

    # Step 1: letterbox pad — paste content at top-left on a black canvas
    padded = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    padded.paste(img, (0, 0))

    # Step 2: normalise bounding box coords into padded-square space
    sequence: List[Tuple] = []
    for child in children:
        # Support both 'label' and 'class' keys found in RICO JSON files
        cls_name = child.get("label", child.get("class", ""))
        if cls_name not in CLASS_TO_IDX:
            continue                                    # skip unknown classes

        x1, y1, x2, y2 = child["bounds"]

        # Two-step normalisation → coords relative to padded square
        x1_n = max(0.0, min(1.0, x1 / max_dim))
        y1_n = max(0.0, min(1.0, y1 / max_dim))
        x2_n = max(0.0, min(1.0, x2 / max_dim))
        y2_n = max(0.0, min(1.0, y2 / max_dim))

        sequence.append((x1_n, y1_n, x2_n, y2_n, CLASS_TO_IDX[cls_name]))

    # Step 3: sort by reading order (top → bottom, left → right)
    sequence.sort(key=lambda t: (t[1], t[0]))           # sort by (y1_norm, x1_norm)

    # Step 4: resize padded image to IMAGE_SIZE × IMAGE_SIZE
    resized = padded.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR)
    image_tensor = _NORMALIZE(resized)                  # [3, IMAGE_SIZE, IMAGE_SIZE]

    return image_tensor, sequence


# ── Dataset ───────────────────────────────────────────────────────────────────

class RICODataset(Dataset):
    """
    PyTorch Dataset for the RICO UI component detection task.

    Args:
        data_dir  : root folder containing images/ and labels/ sub-directories.
        split     : 'train' or 'val'.
        val_split : fraction of unique image stems held out for validation.
        seed      : random seed for the deterministic train/val split.
    """

    def __init__(
        self,
        data_dir:  str   = None,
        split:     str   = "train",
        val_split: float = None,
        seed:      int   = 42,
    ):
        data_dir  = data_dir  or config.DATA_DIR
        val_split = val_split or config.VAL_SPLIT

        images_dir = os.path.join(data_dir, "images")
        labels_dir = os.path.join(data_dir, "labels")

        # Collect stems that have both an image and a label file
        stems = sorted([
            p.stem for p in Path(images_dir).glob("*.jpg")
            if (Path(labels_dir) / f"{p.stem}.json").exists()
        ])

        # Deterministic shuffle → split
        rng = random.Random(seed)
        stems_shuffled = stems.copy()
        rng.shuffle(stems_shuffled)

        n_val    = max(1, int(len(stems_shuffled) * val_split))
        val_set  = set(stems_shuffled[:n_val])
        train_set = set(stems_shuffled[n_val:])
        target   = train_set if split == "train" else val_set

        self.samples = [
            (
                os.path.join(images_dir, f"{s}.jpg"),
                os.path.join(labels_dir, f"{s}.json"),
            )
            for s in stems if s in target
        ]

        if not self.samples:
            raise RuntimeError(f"No samples found for split='{split}' in {data_dir}")

        print(f"RICODataset [{split}]: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, lbl_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        with open(lbl_path, "r") as f:
            label = json.load(f)

        children = label.get("children", [])

        image_tensor, sequence = _letterbox_and_normalise(img, children)

        return {
            "image":    image_tensor,           # FloatTensor [3, IMAGE_SIZE, IMAGE_SIZE]
            "sequence": sequence,               # List[(x1,y1,x2,y2,class_idx)]  variable length
            "length":   len(sequence),          # int
        }


# ── Collate function ──────────────────────────────────────────────────────────

def collate_fn(batch: list) -> dict:
    """
    Pad a list of dataset items to the same sequence length.

    Returns:
        images        : FloatTensor  [B, 3, H, W]
        coords        : FloatTensor  [B, max_len, 4]   — padded with 0.0
        classes       : LongTensor   [B, max_len]       — padded with 0
        padding_mask  : BoolTensor   [B, max_len]       — True at PAD positions
        lengths       : LongTensor   [B]
    """
    images   = torch.stack([item["image"] for item in batch], dim=0)  # [B, 3, H, W]
    lengths  = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    max_len  = int(lengths.max().item())

    B = len(batch)
    coords       = torch.zeros(B, max_len, 4,  dtype=torch.float)
    classes      = torch.zeros(B, max_len,     dtype=torch.long)
    padding_mask = torch.ones(B,  max_len,     dtype=torch.bool)   # default = PAD

    for i, item in enumerate(batch):
        L = item["length"]
        if L == 0:
            continue
        seq = item["sequence"]
        for t, (x1, y1, x2, y2, cls) in enumerate(seq):
            coords[i, t]  = torch.tensor([x1, y1, x2, y2], dtype=torch.float)
            classes[i, t] = cls
        padding_mask[i, :L] = False                  # real positions → not PAD

    return {
        "images":       images,         # [B, 3, H, W]
        "coords":       coords,         # [B, max_len, 4]
        "classes":      classes,        # [B, max_len]
        "padding_mask": padding_mask,   # [B, max_len]  True = PAD
        "lengths":      lengths,        # [B]
    }


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloader(
    split:      str  = "train",
    data_dir:   str  = None,
    val_split:  float = None,
    batch_size: int  = None,
    num_workers: int = None,
    seed:       int  = 42,
):
    from torch.utils.data import DataLoader

    ds = RICODataset(
        data_dir  = data_dir  or config.DATA_DIR,
        split     = split,
        val_split = val_split or config.VAL_SPLIT,
        seed      = seed,
    )
    return DataLoader(
        ds,
        batch_size  = batch_size  or config.BATCH_SIZE,
        shuffle     = (split == "train"),
        num_workers = num_workers or config.NUM_WORKERS,
        pin_memory  = True,
        drop_last   = (split == "train"),
        collate_fn  = collate_fn,
    )


# ── Quick sanity check: python dataset.py ────────────────────────────────────

if __name__ == "__main__":
    loader = get_dataloader(split="val", batch_size=4, num_workers=0)
    batch  = next(iter(loader))
    print("images       :", batch["images"].shape)        # [4, 3, 512, 512]
    print("coords       :", batch["coords"].shape)        # [4, max_len, 4]
    print("classes      :", batch["classes"].shape)       # [4, max_len]
    print("padding_mask :", batch["padding_mask"].shape)  # [4, max_len]
    print("lengths      :", batch["lengths"])
    print("OK")
