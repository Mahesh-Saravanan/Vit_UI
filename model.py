"""
model.py — Autoregressive UI Component Detection Model.

Architecture:
  ┌──────────────────────────────────────────────────────────────────┐
  │  ViTEncoder  (ViT-B/16, image_size=512)                         │
  │   PatchEmbedding → CLS token → PositionalEmbedding              │
  │   → ViTBlock × 12  (MSA + MLP, pre-norm)                        │
  │   → LayerNorm → [B, 1025, 768]   (1024 patches + 1 CLS)         │
  └────────────────────────┬─────────────────────────────────────────┘
                           │  LinearAdapter (vit_dim → d_model)
  ┌────────────────────────▼─────────────────────────────────────────┐
  │  UIDetectionDecoder                                              │
  │   UIElementEmbedding:                                            │
  │     coord_proj  Linear(4 → d_model/2)  ┐                        │
  │     class_emb   Embedding(25 → d_model/2) ┤ cat → Linear → tok  │
  │   bos_emb  (learnable [1, d_model])                              │
  │   pos_emb  Embedding(max_seq_len, d_model)                       │
  │   → DecoderBlock × NUM_LAYERS                                    │
  │       ├─ LN + CausalSelfAttention  + residual                    │
  │       ├─ LN + CrossAttention (← ViT patches)  + residual        │
  │       └─ LN + MLP  + residual                                    │
  │   → LayerNorm                                                    │
  │   ├─ coord_head  Linear(d_model → 4) + Sigmoid  → [B, T+1, 4]  │
  │   ├─ class_head  Linear(d_model → 25)           → [B, T+1, 25]  │
  │   └─ eos_head    Linear(d_model → 1) + Sigmoid  → [B, T+1]      │
  └──────────────────────────────────────────────────────────────────┘

Teacher-forced training:
  Decoder input  = [BOS, e_1, e_2, ..., e_T]           T+1 tokens
  Decoder output = predictions for [e_1, ..., e_T, EOS]  T+1 slots
    • slots 0 .. T-1 : coord + class targets = GT elements 0 .. T-1
    • slot  T        : stop target = 1 (EOS)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ============================================================
#  SECTION 1 · Vision Transformer Encoder  (unchanged from V1)
# ============================================================

class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size:  int = 512,
        patch_size:  int = 16,
        in_channels: int = 3,
        embed_dim:   int = 768,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2   # 1024 for 512px
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.proj(x)         # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)         # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)    # [B, num_patches, embed_dim]
        return x


class ViTAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv       = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj      = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H, D    = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                             # each [B, H, N, D]

        attn = (q @ k.transpose(-2, -1)) * self.scale       # [B, H, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)     # [B, N, C]
        return self.proj(x)


class ViTMLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1  = nn.Linear(embed_dim, hidden)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = ViTAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = ViTMLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    ViT-B/16 encoder.  For image_size=512 and patch_size=16:
        num_patches = (512/16)^2 = 1024
        output shape: [B, 1025, embed_dim]  (1024 patches + 1 CLS token)
    """

    def __init__(
        self,
        image_size:  int   = 512,
        patch_size:  int   = 16,
        in_channels: int   = 3,
        embed_dim:   int   = 768,
        depth:       int   = 12,
        num_heads:   int   = 12,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches      = self.patch_embed.num_patches           # 1024

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, 3, 512, 512]
        B   = pixel_values.shape[0]
        x   = self.patch_embed(pixel_values)               # [B, 1024, D]
        cls = self.cls_token.expand(B, -1, -1)             # [B,    1, D]
        x   = torch.cat([cls, x], dim=1)                   # [B, 1025, D]
        x   = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        return self.norm(x)                                # [B, 1025, D]


# ============================================================
#  SECTION 2 · Cross-Attention  (unchanged from V1)
# ============================================================

class CrossAttention(nn.Module):
    """
    Query  = decoder hidden states  [B, T, C]
    Key/V  = ViT encoder output     [B, 1025, C]
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        decoder_hidden: torch.Tensor,   # [B, T, C]
        encoder_output: torch.Tensor,   # [B, 1025, C]
    ) -> torch.Tensor:
        B, T, C = decoder_hidden.shape
        S       = encoder_output.shape[1]                   # 1025
        H, D    = self.num_heads, self.head_dim

        Q = self.q_proj(decoder_hidden).view(B, T, H, D).transpose(1, 2)   # [B, H, T, D]
        K = self.k_proj(encoder_output).view(B, S, H, D).transpose(1, 2)   # [B, H, S, D]
        V = self.v_proj(encoder_output).view(B, S, H, D).transpose(1, 2)   # [B, H, S, D]

        attn = (Q @ K.transpose(-2, -1)) * self.scale       # [B, H, T, S]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, T, C)   # [B, T, C]
        return self.resid_drop(self.out_proj(out))


# ============================================================
#  SECTION 3 · GPT-2 Style Decoder Blocks  (unchanged from V1)
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with a lower-triangular mask.
    """

    def __init__(
        self,
        hidden_dim:  int,
        num_heads:   int,
        max_seq_len: int   = 128,
        dropout:     float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv        = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)
        self.out        = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D    = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                             # each [B, H, T, D]

        attn = (q @ k.transpose(-2, -1)) * self.scale       # [B, H, T, T]
        attn = attn.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float("-inf")
        )
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.out(x))


class DecoderMLP(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        inner = int(hidden_dim * mlp_ratio)
        self.fc1  = nn.Linear(hidden_dim, inner)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(inner, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DecoderBlock(nn.Module):
    """
    Pre-norm decoder block:  CausalSelfAttn → CrossAttn → MLP.
    """

    def __init__(
        self,
        hidden_dim:  int,
        num_heads:   int,
        max_seq_len: int   = 128,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.ln1        = nn.LayerNorm(hidden_dim)
        self.self_attn  = CausalSelfAttention(hidden_dim, num_heads, max_seq_len, dropout)
        self.ln_cross   = nn.LayerNorm(hidden_dim)
        self.cross_attn = CrossAttention(hidden_dim, num_heads, dropout)
        self.ln2        = nn.LayerNorm(hidden_dim)
        self.mlp        = DecoderMLP(hidden_dim, mlp_ratio, dropout)

    def forward(
        self,
        x:              torch.Tensor,   # [B, T, hidden_dim]
        encoder_output: torch.Tensor,   # [B, 1025, hidden_dim]
    ) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln_cross(x), encoder_output)
        x = x + self.mlp(self.ln2(x))
        return x


# ============================================================
#  SECTION 4 · UI Element Input Embedding  (NEW)
# ============================================================

class UIElementEmbedding(nn.Module):
    """
    Embeds one UI element (x1, y1, x2, y2, class_idx) into d_model dimensions.

    coord_proj  Linear(4 → d_model//2)      continuous bbox
    class_emb   Embedding(num_classes → d_model//2)  discrete label
    fuse        Linear(d_model → d_model)   fused token embedding
    """

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        half = d_model // 2
        self.coord_proj = nn.Linear(4, half)
        self.class_emb  = nn.Embedding(num_classes, half)
        self.fuse       = nn.Linear(d_model, d_model)

    def forward(
        self,
        coords:  torch.Tensor,   # [B, T, 4]  normalised coordinates
        classes: torch.Tensor,   # [B, T]     class indices (long)
    ) -> torch.Tensor:
        coord_feat  = self.coord_proj(coords)       # [B, T, d_model//2]
        class_feat  = self.class_emb(classes)       # [B, T, d_model//2]
        fused       = torch.cat([coord_feat, class_feat], dim=-1)  # [B, T, d_model]
        return self.fuse(fused)                     # [B, T, d_model]


# ============================================================
#  SECTION 5 · UI Detection Decoder  (NEW — replaces GPT2Decoder)
# ============================================================

class UIDetectionDecoder(nn.Module):
    """
    Autoregressive decoder for UI element detection.

    During training (teacher-forcing):
      Input:  [BOS, e_1, ..., e_T]   →  T+1 tokens
      Output: predictions at each of the T+1 positions via three parallel heads.

    Output heads:
      coord_head  → [B, T+1, 4]        normalised bbox predictions (post-Sigmoid)
      class_head  → [B, T+1, num_cls]  class logits (pre-softmax)
      eos_head    → [B, T+1]           stop probability (post-Sigmoid)
    """

    def __init__(
        self,
        d_model:     int,
        num_layers:  int,
        num_heads:   int,
        num_classes: int,
        max_seq_len: int   = 128,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.1,
    ):
        super().__init__()

        # Input embedding for UI elements
        self.elem_emb = UIElementEmbedding(d_model, num_classes)

        # Learnable BOS token  [1, 1, d_model]
        self.bos_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional embedding — position 0 = BOS
        self.pos_emb  = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, max_seq_len, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        # Output heads
        self.coord_head = nn.Linear(d_model, 4)           # + Sigmoid applied in forward
        self.class_head = nn.Linear(d_model, num_classes)
        self.eos_head   = nn.Linear(d_model, 1)           # + Sigmoid applied in forward

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        coords:         torch.Tensor,   # [B, T, 4]     GT element coords (teacher forced)
        classes:        torch.Tensor,   # [B, T]        GT class indices
        encoder_output: torch.Tensor,   # [B, 1025, d_model]
    ) -> tuple:
        B, T  = coords.shape[:2]
        device = coords.device

        # Embed GT elements  [B, T, d_model]
        elem_tokens = self.elem_emb(coords, classes)      # [B, T, d_model]

        # Prepend BOS → decoder input is T+1 tokens
        bos = self.bos_emb.expand(B, -1, -1)              # [B, 1, d_model]
        tokens = torch.cat([bos, elem_tokens], dim=1)     # [B, T+1, d_model]

        # Positional embeddings
        pos    = torch.arange(T + 1, device=device).unsqueeze(0)  # [1, T+1]
        hidden = self.emb_drop(tokens + self.pos_emb(pos))         # [B, T+1, d_model]

        # Transformer blocks
        for block in self.blocks:
            hidden = block(hidden, encoder_output)         # [B, T+1, d_model]

        hidden = self.ln_f(hidden)                         # [B, T+1, d_model]

        # Output heads
        pred_coords = torch.sigmoid(self.coord_head(hidden))   # [B, T+1, 4]
        pred_logits = self.class_head(hidden)                  # [B, T+1, num_classes]
        pred_stop   = self.eos_head(hidden).squeeze(-1)        # [B, T+1]  raw logits — use BCEWithLogits in loss

        return pred_coords, pred_logits, pred_stop

    def decode_step(
        self,
        tokens:         torch.Tensor,   # [1, t, d_model]  current accumulated sequence
        encoder_output: torch.Tensor,   # [1, 1025, d_model]
    ) -> tuple:
        """Single autoregressive decode step.  Returns heads at the last position."""
        device = tokens.device
        T_cur  = tokens.shape[1]

        pos    = torch.arange(T_cur, device=device).unsqueeze(0)
        hidden = self.emb_drop(tokens + self.pos_emb(pos))

        for block in self.blocks:
            hidden = block(hidden, encoder_output)

        hidden = self.ln_f(hidden)
        last   = hidden[:, -1:, :]                         # [1, 1, d_model]

        coords  = torch.sigmoid(self.coord_head(last)).squeeze(1)    # [1, 4]
        logits  = self.class_head(last).squeeze(1)                    # [1, num_classes]
        stop    = torch.sigmoid(self.eos_head(last)).squeeze()        # scalar

        return coords, logits, stop


# ============================================================
#  SECTION 6 · Full VisionUIDetector  (NEW — replaces VisionGPT2Model)
# ============================================================

class VisionUIDetector(nn.Module):
    """
    End-to-end autoregressive UI component detection model.

    Image → ViTEncoder → LinearAdapter → UIDetectionDecoder → heads
    """

    def __init__(
        self,
        image_size:  int   = None,
        patch_size:  int   = None,
        vit_dim:     int   = None,
        vit_depth:   int   = None,
        vit_heads:   int   = None,
        d_model:     int   = None,
        num_layers:  int   = None,
        num_heads:   int   = None,
        num_classes: int   = None,
        max_seq_len: int   = None,
        mlp_ratio:   float = None,
        dropout:     float = None,
    ):
        super().__init__()

        # Resolve defaults from config
        image_size  = image_size  or config.IMAGE_SIZE
        patch_size  = patch_size  or config.PATCH_SIZE
        vit_dim     = vit_dim     or config.VIT_DIM
        vit_depth   = vit_depth   or config.VIT_DEPTH
        vit_heads   = vit_heads   or config.VIT_HEADS
        d_model     = d_model     or config.D_MODEL
        num_layers  = num_layers  or config.NUM_LAYERS
        num_heads   = num_heads   or config.NUM_HEADS
        num_classes = num_classes or config.NUM_CLASSES
        max_seq_len = max_seq_len or config.MAX_SEQ_LEN
        mlp_ratio   = mlp_ratio   or config.MLP_RATIO
        dropout     = dropout     or config.DROPOUT

        self.encoder = ViTEncoder(
            image_size=image_size, patch_size=patch_size,
            embed_dim=vit_dim, depth=vit_depth, num_heads=vit_heads,
            mlp_ratio=mlp_ratio, dropout=dropout,
        )

        # Project ViT output to decoder dim (identity-like when dims match)
        self.adapter = (
            nn.Linear(vit_dim, d_model)
            if vit_dim != d_model
            else nn.Identity()
        )

        self.decoder = UIDetectionDecoder(
            d_model=d_model, num_layers=num_layers, num_heads=num_heads,
            num_classes=num_classes, max_seq_len=max_seq_len,
            mlp_ratio=mlp_ratio, dropout=dropout,
        )

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values [B, 3, H, W] → encoder output [B, 1025, d_model]"""
        return self.adapter(self.encoder(pixel_values))  # [B, 1025, d_model]

    def forward(
        self,
        pixel_values: torch.Tensor,   # [B, 3, 512, 512]
        coords:       torch.Tensor,   # [B, T, 4]
        classes:      torch.Tensor,   # [B, T]
    ) -> tuple:
        """
        Teacher-forced forward pass.
        Returns:
            pred_coords : [B, T+1, 4]          bbox predictions (sigmoid applied)
            pred_logits : [B, T+1, num_classes] class logits
            pred_stop   : [B, T+1]              stop probabilities (sigmoid applied)
        """
        enc_out = self.encode_image(pixel_values)           # [B, 1025, d_model]
        return self.decoder(coords, classes, enc_out)
