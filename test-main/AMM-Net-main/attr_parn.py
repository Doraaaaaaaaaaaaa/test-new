"""
PARN-style Attribute Encoder for AADB 11 aesthetic attributes.

Reference: Kong et al., "Photo Aesthetics Ranking Network with Attributes and
Content Adaptation", ECCV 2016.

Architecture (matches AMM-Net.pt img_attr module):
  ResNet-50 backbone → g (B, 2048)
  Shared bottleneck:
    fc1_1(2048→256) → bn1_1 → PReLU
    → fc2_1(256→64) → PReLU
    → fc3_1(64→11)          [11 raw attribute scores]
  11 attribute token projectors: a_i = Linear(2049, out_dim) + LayerNorm + ReLU
  Output: Fa = stack(a_0..a_10, dim=1)  → (B, 11, out_dim)

Key-name layout in AMM-Net.pt (prefix "img_attr."):
  img_attr.conv1.*  img_attr.bn1.*  img_attr.layer{1-4}.*  img_attr.avgpool.*
  img_attr.fc1_1.*  img_attr.bn1_1.*  img_attr.relu1_1.*
  img_attr.fc2_1.*  img_attr.relu2_1.*
  img_attr.fc3_1.*
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

AADB_ATTR_NAMES = [
    "interesting_content",
    "object_emphasis",
    "good_lighting",
    "color_harmony",
    "vivid_color",
    "shallow_dof",
    "motion_blur",
    "rule_of_thirds",
    "balancing_element",
    "repetition",
    "symmetry",
]

NUM_ATTRS = 11


class PARNAttributeEncoder(nn.Module):
    """
    Encodes an image into 11 aesthetic attribute tokens.

    Module names are kept identical to the AMM-Net.pt ``img_attr.*`` subtree
    so that pretrained weights can be loaded by stripping the ``img_attr.``
    prefix and calling load_state_dict(strict=False).

    Args:
        out_dim: dimension of each attribute token (default 2048 = D).
        pretrained_path: path to either
            * the full AMM-Net.pt (flat OrderedDict with ``img_attr.*`` keys), or
            * a standalone state_dict / {"model": state_dict} checkpoint.
          If None, weights are randomly initialised.
        freeze: if True, backbone + bottleneck parameters are frozen after
          loading (attr_projs are always trainable).
    """

    def __init__(self, out_dim: int = 2048, pretrained_path: str = None,
                 freeze: bool = False):
        super().__init__()

        resnet = tv_models.resnet50(weights=None)

        # ── Backbone: direct attributes match AMM-Net.pt key names ───────────
        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.avgpool = resnet.avgpool

        # ── Shared bottleneck (matches AMM-Net.pt img_attr.fc*) ──────────────
        self.fc1_1   = nn.Linear(2048, 256)
        self.bn1_1   = nn.BatchNorm1d(256)
        self.relu1_1 = nn.PReLU()
        self.fc2_1   = nn.Linear(256, 64)
        self.relu2_1 = nn.PReLU()
        self.fc3_1   = nn.Linear(64, NUM_ATTRS)

        # ── Per-attribute token projectors (not in pretrained checkpoint) ─────
        self.attr_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2048 + 1, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(NUM_ATTRS)
        ])

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        if freeze:
            # Freeze backbone + bottleneck; keep attr_projs trainable
            for name, p in self.named_parameters():
                if not name.startswith("attr_projs"):
                    p.requires_grad = False

    # ── Weight loading ────────────────────────────────────────────────────────

    def _load_pretrained(self, path: str):
        ckpt = torch.load(path, map_location="cpu")

        # Unwrap {"model": ...} wrapper if present
        if isinstance(ckpt, dict) and "model" in ckpt and not any(
            k.startswith("img_attr.") for k in ckpt
        ):
            ckpt = ckpt["model"]

        # Full AMM-Net.pt: flat OrderedDict with "img_attr.*" keys
        if any(k.startswith("img_attr.") for k in ckpt):
            state = {
                k[len("img_attr."):]: v
                for k, v in ckpt.items()
                if k.startswith("img_attr.")
            }
            missing, unexpected = self.load_state_dict(state, strict=False)
            # attr_projs keys will always be "missing" since they're not in ckpt
            backbone_missing = [k for k in missing if not k.startswith("attr_projs")]
            if backbone_missing:
                print(f"PARNAttributeEncoder: missing backbone keys: {backbone_missing}")
            print(f"PARNAttributeEncoder: loaded img_attr weights from {path}")
        else:
            # Standalone state_dict (e.g. a separately saved PARN checkpoint)
            missing, unexpected = self.load_state_dict(ckpt, strict=False)
            backbone_missing = [k for k in missing if not k.startswith("attr_projs")]
            if backbone_missing:
                print(f"PARNAttributeEncoder: missing backbone keys: {backbone_missing}")
            print(f"PARNAttributeEncoder: loaded weights from {path}")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward_from_cache(self, g: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Skip backbone and bottleneck; only run trainable attr_projs.
        Use this when g and scores have been precomputed and cached.

        Args:
            g:      (B, 2048)  cached ResNet-50 global features
            scores: (B, 11)    cached attribute scores from bottleneck
        Returns:
            Fa: (B, NUM_ATTRS, out_dim)
        """
        tokens = []
        for i in range(NUM_ATTRS):
            z_i = scores[:, i:i + 1]
            a_i = self.attr_projs[i](torch.cat([g, z_i], dim=-1))
            tokens.append(a_i.unsqueeze(1))
        return torch.cat(tokens, dim=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) — ImageNet-normalised; typically 224×224.
        Returns:
            Fa: (B, NUM_ATTRS, out_dim)
        """
        # ResNet-50 feature extraction
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        g = x.flatten(1)                             # (B, 2048)

        # Shared bottleneck → 11 attribute scores
        h = self.relu1_1(self.bn1_1(self.fc1_1(g))) # (B, 256)
        h = self.relu2_1(self.fc2_1(h))             # (B, 64)
        scores = self.fc3_1(h)                       # (B, 11)

        # Per-attribute token generation
        tokens = []
        for i in range(NUM_ATTRS):
            z_i = scores[:, i:i + 1]                          # (B, 1)
            a_i = self.attr_projs[i](torch.cat([g, z_i], dim=-1))  # (B, out_dim)
            tokens.append(a_i.unsqueeze(1))                   # (B, 1, out_dim)

        return torch.cat(tokens, dim=1)              # (B, 11, out_dim)
