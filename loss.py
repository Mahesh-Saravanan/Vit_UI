"""
loss.py — Combined detection loss for the UI component detection model.

L_total = λ1 * L_coord + λ2 * L_class + λ3 * L_stop

  L_coord — SmoothL1Loss between predicted and GT normalised boxes.
            Applied only at non-PAD positions (real element predictions).

  L_class — CrossEntropyLoss between class logits and GT class indices.
            Applied only at non-PAD positions.

  L_stop  — BCELoss on the EOS head.
            Target = 0 for all real-element output positions,
            Target = 1 at the EOS output position (index == length).
            PAD positions beyond EOS are excluded.

Alignment of positions (T = max sequence length in the batch):
  Decoder input  : [BOS, e_1, ..., e_T]   (T+1 tokens)
  Decoder output : [pred_0, ..., pred_T]   (T+1 predictions)
    pred_t  (0 ≤ t < T) → predicts the (t+1)-th GT element  = targets[:, t, :]
    pred_T                → EOS signal (stop = 1)
"""

import torch
import torch.nn.functional as F

import config


def compute_loss(
    pred_coords:   torch.Tensor,    # [B, T+1, 4]
    pred_logits:   torch.Tensor,    # [B, T+1, num_classes]
    pred_stop:     torch.Tensor,    # [B, T+1]
    target_coords: torch.Tensor,    # [B, T, 4]
    target_classes: torch.Tensor,   # [B, T]         long
    lengths:       torch.Tensor,    # [B]             long  — real element count per sample
    padding_mask:  torch.Tensor,    # [B, T]          bool  — True at PAD positions
    lambda_coord:  float = None,
    lambda_class:  float = None,
    lambda_stop:   float = None,
) -> tuple:
    """
    Returns (loss_total, loss_coord, loss_class, loss_stop) — all scalar tensors.
    """
    lambda_coord = lambda_coord if lambda_coord is not None else config.LAMBDA_COORD
    lambda_class = lambda_class if lambda_class is not None else config.LAMBDA_CLASS
    lambda_stop  = lambda_stop  if lambda_stop  is not None else config.LAMBDA_STOP

    B  = target_coords.shape[0]
    T  = target_coords.shape[1]   # max sequence length
    device = pred_coords.device

    # ── valid mask: [B, T]  True where we compute coord + class loss ──────────
    valid_mask = ~padding_mask                               # [B, T]  True = real element

    # ── L_coord ───────────────────────────────────────────────────────────────
    # pred_coords[:, :T, :] are predictions for elements 0..T-1
    if valid_mask.any():
        pred_c = pred_coords[:, :T, :]                      # [B, T, 4]
        coord_loss = F.smooth_l1_loss(
            pred_c[valid_mask],                             # [N_valid, 4]
            target_coords[valid_mask],                      # [N_valid, 4]
            reduction="mean",
        )
    else:
        coord_loss = torch.tensor(0.0, device=device)

    # ── L_class ───────────────────────────────────────────────────────────────
    if valid_mask.any():
        pred_l = pred_logits[:, :T, :]                      # [B, T, num_classes]
        class_loss = F.cross_entropy(
            pred_l[valid_mask],                             # [N_valid, num_classes]
            target_classes[valid_mask],                     # [N_valid]
            reduction="mean",
        )
    else:
        class_loss = torch.tensor(0.0, device=device)

    # ── L_stop ────────────────────────────────────────────────────────────────
    # Build stop target [B, T+1] and a mask covering positions 0..length_i
    stop_target = torch.zeros(B, T + 1, device=device)      # [B, T+1]  default = 0
    stop_mask   = torch.zeros(B, T + 1, dtype=torch.bool, device=device)

    for i in range(B):
        L = int(lengths[i].item())
        # Stop = 1 at the EOS position (right after the last real element)
        if L <= T:
            stop_target[i, L] = 1.0
        # Compute stop loss for positions 0..L (inclusive of EOS position)
        stop_mask[i, : L + 1] = True

    if stop_mask.any():
        stop_loss = F.binary_cross_entropy_with_logits(
            pred_stop[stop_mask],                           # [N_stop]  raw logits
            stop_target[stop_mask],                         # [N_stop]
            reduction="mean",
        )
    else:
        stop_loss = torch.tensor(0.0, device=device)

    # ── Total loss ────────────────────────────────────────────────────────────
    loss_total = (
        lambda_coord * coord_loss
        + lambda_class * class_loss
        + lambda_stop  * stop_loss
    )

    return loss_total, coord_loss, class_loss, stop_loss
