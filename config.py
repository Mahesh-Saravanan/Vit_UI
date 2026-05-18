"""
config.py — all hyperparameters for the UI component detection model.
Edit this file to tune the model; every other script imports from here.
"""

# ── UI classes ────────────────────────────────────────────────────────────────
UI_CLASSES = [
    'Advertisement', 'Background Image', 'Bottom Navigation', 'Button Bar',
    'Card', 'Checkbox', 'Date Picker', 'Drawer', 'Icon', 'Image', 'Input',
    'List Item', 'Map View', 'Modal', 'Multi-Tab', 'Number Stepper',
    'On/Off Switch', 'Pager Indicator', 'Radio Button', 'Slider', 'Text',
    'Text Button', 'Toolbar', 'Video', 'Web View',
]
NUM_CLASSES = len(UI_CLASSES)   # 25 (indices 0–24)

# ── ViT encoder (ViT-B/16 at 512×512) ────────────────────────────────────────
IMAGE_SIZE  = 512    # both dimensions; images are letterbox-padded then resized
PATCH_SIZE  = 16     # 512 / 16 = 32 patches per side → 1024 patch tokens + 1 CLS
VIT_DIM     = 768
VIT_DEPTH   = 12
VIT_HEADS   = 12

# ── Autoregressive decoder ────────────────────────────────────────────────────
D_MODEL     = 768    # must equal VIT_DIM if no adapter projection is desired
NUM_LAYERS  = 6      # number of DecoderBlocks (keep lower than ViT depth for speed)
NUM_HEADS   = 12
MLP_RATIO   = 4.0
DROPOUT     = 0.1
MAX_SEQ_LEN = 512    # BOS + up to 511 elements; RICO 60k has screens with 264+ components

# ── Loss weights ──────────────────────────────────────────────────────────────
LAMBDA_COORD = 1.0
LAMBDA_CLASS = 1.0
LAMBDA_STOP  = 1.0

# ── Optimiser ─────────────────────────────────────────────────────────────────
LR           = 1e-4
WEIGHT_DECAY = 1e-2

# ── Training schedule ─────────────────────────────────────────────────────────
EPOCHS         = 50
FREEZE_EPOCHS  = 5    # keep ViT encoder frozen for the first N epochs
UNFREEZE_BLOCKS = 4   # unfreeze only the last N ViT blocks (-1 = unfreeze all 12)
GRAD_CHECKPOINT = True  # gradient checkpointing on ViT when unfrozen (~40% less activation memory)
BATCH_SIZE     = 8
WARMUP_RATIO   = 0.05 # fraction of total steps used for linear LR warmup
GRAD_CLIP      = 1.0
NUM_WORKERS    = 4

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
DATA_DIR       = "Data"
VAL_SPLIT      = 0.1   # fraction of images held out for validation

# ── Inference ─────────────────────────────────────────────────────────────────
MAX_ELEMENTS  = 50    # autoregressive decode cap (hard stop regardless of EOS)
STOP_THRESH   = 0.5   # stop probability threshold
