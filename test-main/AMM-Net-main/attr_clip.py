import torch
import torch.nn as nn
import torch.nn.functional as F

import clip

AADB_PROMPTS_11 = [
    "a photo with interesting content",
    "a photo with clear object emphasis",
    "a photo with good lighting",
    "a photo with good color harmony",
    "a photo with vivid color",
    "a photo with shallow depth of field",
    "a photo with motion blur",
    "a photo following rule of thirds",
    "a photo with balanced elements",
    "a photo with repetition patterns",
    "a photo with symmetry",
]


class RobustClipAttributeEncoder(nn.Module):
    """
    CLIP prompt-bank attribute encoder.
    Output: Fa (B, m, out_dim), default out_dim=2048 for AMM-Net.

    Supports two input modes:
    1) img_clip: (B, 3, 224, 224) CLIP-normalized images
    2) img_clip: (B, d_clip) precomputed CLIP image embeddings
    """

    def __init__(
        self,
        prompts=AADB_PROMPTS_11,
        out_dim=2048,
        clip_name="ViT-B/16",
        freeze_clip=True,
        temperature=0.07,
        device=None,
        download_root=None,
    ):
        super().__init__()
        self.temperature = temperature
        self.freeze_clip = freeze_clip

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.clip_model, _ = clip.load(
            clip_name,
            device=self.device,
            download_root=download_root
        )
        self.clip_model.eval()

        if freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        with torch.no_grad():
            tokens = clip.tokenize(prompts).to(self.device)   # (m, 77)
            t = self.clip_model.encode_text(tokens)           # (m, d_clip)
            t = F.normalize(t.float(), dim=-1)

        self.register_buffer("prompt_emb", t)  # (m, d_clip)

        d_clip = t.shape[-1]
        self.d_clip = d_clip

        self.proj_T = nn.Linear(d_clip, out_dim)
        self.proj_V = nn.Linear(d_clip, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def _encode_or_accept_feature(self, img_clip: torch.Tensor) -> torch.Tensor:
        """
        Accept either:
        - image tensor: (B, 3, 224, 224)
        - precomputed CLIP embedding: (B, d_clip)
        Return normalized image embedding v: (B, d_clip)
        """
        img_clip = img_clip.to(self.prompt_emb.device)

        # Case 1: precomputed CLIP feature
        if img_clip.dim() == 2 and img_clip.size(-1) == self.d_clip:
            v = img_clip.float()
            v = F.normalize(v, dim=-1)
            return v

        # Case 2: CLIP-normalized image tensor
        if img_clip.dim() == 4 and img_clip.size(1) == 3:
            if self.freeze_clip:
                with torch.no_grad():
                    v = self.clip_model.encode_image(img_clip)
            else:
                v = self.clip_model.encode_image(img_clip)

            v = F.normalize(v.float(), dim=-1)
            return v

        raise ValueError(
            f"Unsupported img_clip shape {tuple(img_clip.shape)}. "
            f"Expected (B,3,224,224) image tensor or (B,{self.d_clip}) CLIP feature."
        )

    def forward(self, img_clip: torch.Tensor, return_weights: bool = False):
        """
        Returns:
        - Fa: (B, m, 2048)
        - optionally w: (B, m)
        """
        v = self._encode_or_accept_feature(img_clip)  # (B, d_clip)

        logits = (v @ self.prompt_emb.t()) / self.temperature  # (B, m)
        w = torch.softmax(logits, dim=-1)

        T_proj = self.proj_T(self.prompt_emb)                 # (m, out_dim)
        T_proj = T_proj.unsqueeze(0).expand(v.size(0), -1, -1)

        V_proj = self.proj_V(v)                               # (B, out_dim)
        V_gated = V_proj.unsqueeze(1) * w.unsqueeze(-1)       # (B, m, out_dim)

        Fa = self.layer_norm(T_proj + V_gated)

        if return_weights:
            return Fa, w
        return Fa