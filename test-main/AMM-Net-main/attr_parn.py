"""
PARN-style Attribute Encoder for AADB 11 aesthetic attributes.

Reference: Kong et al., "Photo Aesthetics Ranking Network with Attributes and
Content Adaptation", ECCV 2016.

Architecture:
  ResNet-50 shared backbone → g (B, 2048)
  11 attribute score heads: z_i = Linear(2048, 1)
  11 attribute token projectors: a_i = Linear(2049, out_dim) + LayerNorm + ReLU
  Output: Fa = stack(a_0..a_10, dim=1)  → (B, 11, out_dim)
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

    Args:
        out_dim: dimension of each attribute token (default 2048 = D).
        pretrained_path: optional path to a saved state_dict or checkpoint dict
            with key "model".  If None, weights are randomly initialised.
        freeze: if True, all parameters are frozen after loading.
    """

    def __init__(self, out_dim: int = 2048, pretrained_path: str = None,
                 freeze: bool = False):
        super().__init__()

        resnet = tv_models.resnet50(weights=None)
        # Remove the final avgpool and fc → keep spatial feature extractor
        # children: conv1,bn1,relu,maxpool, layer1,layer2,layer3,layer4, avgpool, fc
        self.shared = nn.Sequential(*list(resnet.children())[:-1])  # (B,2048,1,1)

        self.attr_heads = nn.ModuleList([
            nn.Linear(2048, 1) for _ in range(NUM_ATTRS)
        ])
        self.attr_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2048 + 1, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(NUM_ATTRS)
        ])

        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.load_state_dict(state, strict=False)
            print(f"PARNAttributeEncoder: loaded pretrained weights from {pretrained_path}")

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) — ImageNet-normalised; typically 224×224.
        Returns:
            Fa: (B, NUM_ATTRS, out_dim)
        """
        g = self.shared(image).flatten(1)   # (B, 2048)

        attr_tokens = []
        for i in range(NUM_ATTRS):
            z_i = self.attr_heads[i](g)                          # (B, 1)
            a_i = self.attr_projs[i](torch.cat([g, z_i], dim=-1))  # (B, out_dim)
            attr_tokens.append(a_i.unsqueeze(1))                 # (B, 1, out_dim)

        Fa = torch.cat(attr_tokens, dim=1)  # (B, 11, out_dim)
        return Fa
