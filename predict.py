"""
predict.py — Autoregressive inference for the UI component detection model.

Usage:
    python predict.py path/to/screenshot.jpg
    python predict.py path/to/screenshot.jpg --checkpoint checkpoints/best.pt
    python predict.py path/to/screenshot.jpg --max_elements 30

Returns a list of dicts:
    [{'class': 'Toolbar',   'bounds': [0, 84, 1440, 280]},
     {'class': 'Card',      'bounds': [28, 280, 1412, 674]},
     ...]
Bounds are in the original image's pixel coordinate space.

Denormalisation note:
  Model coords are normalised to the letterbox-padded square space:
      coord_norm = coord_pixel / max_dim   where max_dim = max(orig_w, orig_h)
  Denormalise with:
      coord_pixel = coord_norm * max_dim
  Since content was pasted at top-left (0, 0), coord_pixel equals the original
  pixel value directly — no offset correction needed.
"""

import argparse
import torch
from PIL import Image
from torchvision import transforms

import config
from model import VisionUIDetector
from dataset import CLASS_TO_IDX, IDX_TO_CLASS, IMAGENET_MEAN, IMAGENET_STD


# ── Image preprocessing ───────────────────────────────────────────────────────

def preprocess_image(image_path: str) -> tuple:
    """
    Letterbox-pad, resize to IMAGE_SIZE × IMAGE_SIZE, normalise.

    Returns:
        tensor  : FloatTensor [1, 3, IMAGE_SIZE, IMAGE_SIZE]
        orig_w  : int  original image width
        orig_h  : int  original image height
        max_dim : int  max(orig_w, orig_h)  — denormalisation scale
    """
    img    = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    max_dim = max(orig_w, orig_h)

    # Letterbox pad (same as dataset.py)
    padded = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    padded.paste(img, (0, 0))

    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(padded).unsqueeze(0)   # [1, 3, IMAGE_SIZE, IMAGE_SIZE]
    return tensor, orig_w, orig_h, max_dim


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> VisionUIDetector:
    model = VisionUIDetector().to(device)
    from utils import load_checkpoint
    load_checkpoint(model, checkpoint_path, device)
    model.eval()
    return model


# ── Autoregressive prediction ─────────────────────────────────────────────────

@torch.no_grad()
def predict(
    image_path:      str,
    model:           VisionUIDetector,
    device:          torch.device = None,
    max_elements:    int   = None,
    stop_threshold:  float = None,
) -> list:
    """
    Autoregressively detect all UI components in a screenshot.

    Args:
        image_path     : path to the input image.
        model          : loaded VisionUIDetector in eval mode.
        device         : inference device (auto-detected if None).
        max_elements   : hard cap on the number of decoded elements.
        stop_threshold : stop when EOS probability exceeds this value.

    Returns:
        List of {'class': str, 'bounds': [x1, y1, x2, y2]} in original pixel coords.
    """
    if device is None:
        device = next(model.parameters()).device
    max_elements   = max_elements   or config.MAX_ELEMENTS
    stop_threshold = stop_threshold or config.STOP_THRESH

    model.eval()

    # Step 1: Load and preprocess
    image_tensor, orig_w, orig_h, max_dim = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)                 # [1, 3, H, W]

    # Step 2: Encode image once
    enc_out = model.encode_image(image_tensor)             # [1, 1025, d_model]

    # Step 3: Start decoder with BOS token
    decoder  = model.decoder
    bos      = decoder.bos_emb.expand(1, -1, -1)          # [1, 1, d_model]
    tokens   = bos                                         # [1, 1, d_model]

    results = []

    for step in range(max_elements):
        # Step 4: Run one decode step (using accumulated token sequence)
        coords_pred, logits_pred, stop_prob = decoder.decode_step(tokens, enc_out)
        # coords_pred : [1, 4]   normalised [0, 1]
        # logits_pred : [1, 25]
        # stop_prob   : scalar

        # Step 5: Check stop condition
        if stop_prob.item() > stop_threshold:
            break

        # Predicted class (argmax)
        cls_idx  = int(logits_pred.argmax(dim=-1).item())
        cls_name = IDX_TO_CLASS.get(cls_idx, f"unknown_{cls_idx}")

        # Normalised coords
        x1_n, y1_n, x2_n, y2_n = coords_pred[0].tolist()

        # Step 7: Denormalise to original pixel space
        x1 = round(x1_n * max_dim)
        y1 = round(y1_n * max_dim)
        x2 = round(x2_n * max_dim)
        y2 = round(y2_n * max_dim)

        results.append({"class": cls_name, "bounds": [x1, y1, x2, y2]})

        # Step 6: Feed prediction back as next decoder input (autoregressive)
        coords_t  = coords_pred.unsqueeze(1)               # [1, 1, 4]
        classes_t = torch.tensor([[cls_idx]], device=device, dtype=torch.long)  # [1, 1]

        next_emb = decoder.elem_emb(coords_t, classes_t)  # [1, 1, d_model]

        # Positional embedding for the new token
        pos_idx  = torch.tensor([[tokens.shape[1]]], device=device)
        next_tok = next_emb + decoder.pos_emb(pos_idx)    # [1, 1, d_model]

        tokens = torch.cat([tokens, next_tok], dim=1)      # [1, t+1, d_model]

    return results


# ── Visualisation ────────────────────────────────────────────────────────────

# One distinct colour per UI class (RGB, 0-255)
_PALETTE = [
    (230, 25,  75),  (60,  180, 75),  (255, 225, 25),  (0,   130, 200),
    (245, 130, 48),  (145, 30,  180), (70,  240, 240),  (240, 50,  230),
    (210, 245, 60),  (250, 190, 212), (0,   128, 128),  (220, 190, 255),
    (170, 110, 40),  (255, 250, 200), (128, 0,   0),    (170, 255, 195),
    (128, 128, 0),   (255, 215, 180), (0,   0,   128),  (128, 128, 128),
    (255, 255, 255), (0,   0,   0),   (255, 0,   0),    (0,   255, 0),
    (0,   0,   255),
]

def _class_colour(cls_name: str) -> tuple:
    idx = list(config.UI_CLASSES).index(cls_name) if cls_name in config.UI_CLASSES else 0
    return _PALETTE[idx % len(_PALETTE)]


def visualise(
    image_path: str,
    detections: list,
    save_path:  str  = None,
    show:       bool = True,
    box_alpha:  float = 0.35,
    font_scale: float = 0.55,
) -> None:
    """
    Draw semi-transparent bounding boxes + class labels on the original image.

    Args:
        image_path  : path to the original (unprocessed) screenshot.
        detections  : list of {'class': str, 'bounds': [x1,y1,x2,y2]} dicts
                      in original pixel coordinates.
        save_path   : if given, save the annotated image here (PNG/JPG).
        show        : if True, open a window with plt.show().
        box_alpha   : opacity of the filled rectangle (0 = invisible, 1 = solid).
        font_scale  : matplotlib font size for labels.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    fig, ax = plt.subplots(1, 1, figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img)
    ax.axis("off")

    legend_handles = {}

    for det in detections:
        cls_name        = det["class"]
        x1, y1, x2, y2 = det["bounds"]
        bw, bh          = x2 - x1, y2 - y1
        r, g, b         = _class_colour(cls_name)
        colour_norm     = (r / 255, g / 255, b / 255)

        # Semi-transparent filled rectangle
        rect = mpatches.FancyBboxPatch(
            (x1, y1), bw, bh,
            boxstyle="square,pad=0",
            linewidth=1.5,
            edgecolor=colour_norm,
            facecolor=(*colour_norm, box_alpha),
        )
        ax.add_patch(rect)

        # Label: white text on a coloured background pill
        ax.text(
            x1 + 4, y1 + 4,
            cls_name,
            fontsize=max(6, font_scale * 72 * min(w, h) / 1000),
            color="white",
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor=colour_norm,
                alpha=0.85,
                edgecolor="none",
            ),
        )

        # Collect for legend (one entry per unique class)
        if cls_name not in legend_handles:
            legend_handles[cls_name] = mpatches.Patch(
                facecolor=colour_norm, edgecolor=colour_norm, label=cls_name
            )

    if legend_handles:
        ax.legend(
            handles=list(legend_handles.values()),
            loc="upper right",
            fontsize=7,
            framealpha=0.7,
            ncol=max(1, len(legend_handles) // 12),
        )

    plt.title(f"{len(detections)} UI component(s) detected", fontsize=10, pad=6)
    plt.tight_layout(pad=0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Annotated image saved → {save_path}")

    if show:
        plt.show()

    plt.close(fig)


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="UI Component Detection — Inference")
    parser.add_argument("image", type=str, help="Path to the input screenshot")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pt",
        help="Path to model checkpoint (default: checkpoints/best.pt)",
    )
    parser.add_argument(
        "--max_elements", type=int, default=config.MAX_ELEMENTS,
        help=f"Maximum elements to decode (default: {config.MAX_ELEMENTS})",
    )
    parser.add_argument(
        "--stop_threshold", type=float, default=config.STOP_THRESH,
        help=f"EOS probability threshold (default: {config.STOP_THRESH})",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda' or 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--save", type=str, default=None, metavar="PATH",
        help="Also save the annotated image to this path (e.g. out.png)",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not open the interactive plot window (useful on headless servers)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Image      : {args.image}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")

    model = load_model(args.checkpoint, device)

    detections = predict(
        image_path=args.image,
        model=model,
        device=device,
        max_elements=args.max_elements,
        stop_threshold=args.stop_threshold,
    )

    print(f"\nDetected {len(detections)} UI component(s):")
    for i, det in enumerate(detections, 1):
        print(f"  {i:3d}.  {det['class']:<22}  bounds={det['bounds']}")

    visualise(
        image_path=args.image,
        detections=detections,
        save_path=args.save,
        show=not args.no_show,
    )
