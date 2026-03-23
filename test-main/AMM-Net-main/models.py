"""
AADB Attribute-Prior Driven Hierarchical Cross-Modal Dual-State Reasoning Network.

Modules
-------
EncoderText              : Hierarchical BERT encoder (4 groups × 3 layers → D each)
HierarchicalCrossModalFusion : Level-wise cross-attention between Swin stages & BERT groups
DualStateReasoning       : Attribute-conditioned GRU dual-state reasoning (3 hops)
catNet                   : Full model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from attr_parn import PARNAttributeEncoder
from swin_transformer import swin_base_patch4_window7_224_in22k

# ── Shared hyper-parameters ──────────────────────────────────────────────────
D = 2048           # unified hidden dimension
NUM_HEADS = 8      # attention heads (D must be divisible by NUM_HEADS)
NUM_HOPS = 3       # reasoning hops
SWIN_DIMS = [256, 512, 1024, 1024]   # Swin-Base output dims for 448×448 input
VISUAL_POOL = 196  # pool every visual stage to this many tokens
# ─────────────────────────────────────────────────────────────────────────────


class EncoderText(nn.Module):
    """
    Hierarchical BERT text encoder.

    BERT's 12 transformer layers are split into 4 groups of 3.  Within each
    group a learnable softmax-weighted sum produces a single (B, seq, 768)
    representation, which is then projected to (B, seq, D).

    Returns:
        list of 4 tensors, each (B, seq_len, D)
    """

    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.num_groups = 4

        # Learnable intra-group layer weights (3 layers per group)
        self.layer_weights = nn.ParameterList([
            nn.Parameter(torch.ones(3) / 3.0)
            for _ in range(self.num_groups)
        ])

        # Per-group projection: 768 → D
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Linear(768, D), nn.LayerNorm(D))
            for _ in range(self.num_groups)
        ])

    def forward(self, text, attention_mask=None):
        out = self.bert(
            input_ids=text,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # hidden_states[0] = embedding; hidden_states[1..12] = transformer layers
        hidden = out.hidden_states[1:]   # tuple of 12 × (B, seq, 768)

        group_outputs = []
        for g in range(self.num_groups):
            layers = hidden[g * 3: (g + 1) * 3]          # 3 tensors
            w = F.softmax(self.layer_weights[g], dim=0)   # (3,)
            fused = sum(w[i] * layers[i] for i in range(3))  # (B, seq, 768)
            group_outputs.append(self.proj[g](fused))         # (B, seq, D)

        return group_outputs   # list of 4 × (B, seq, D)


class HierarchicalCrossModalFusion(nn.Module):
    """
    Level-wise bidirectional cross-modal fusion.

    For each level i (0..3):
      1. Project i-th Swin stage output to D.
      2. Adaptively pool to VISUAL_POOL tokens.
      3. Cross-attend: visual queries ← text keys/values  → Mv_i
      4. Cross-attend: text queries   ← visual keys/values → Mt_i
      5. Gated residual fusion.

    Args:
        stage_outputs : list of 4 Swin tensors
                        [(B,3136,256),(B,784,512),(B,196,1024),(B,196,1024)]
        text_groups   : list of 4 BERT tensors, each (B, seq, D)
        text_mask     : (B, seq) — 1 for real tokens, 0 for padding

    Returns:
        Mv : list of 4 × (B, VISUAL_POOL, D)
        Mt : list of 4 × (B, seq, D)
    """

    def __init__(self):
        super().__init__()

        self.v_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(SWIN_DIMS[i], D), nn.LayerNorm(D))
            for i in range(4)
        ])

        # Cross-attention: visual (query) ← text (key/value)
        self.cross_v = nn.ModuleList([
            nn.MultiheadAttention(D, NUM_HEADS, batch_first=True, dropout=0.1)
            for _ in range(4)
        ])

        # Cross-attention: text (query) ← visual (key/value)
        self.cross_t = nn.ModuleList([
            nn.MultiheadAttention(D, NUM_HEADS, batch_first=True, dropout=0.1)
            for _ in range(4)
        ])

        # Gating for visual fusion: concat(v, v_attn, v−v_attn) → gate scalar in [0,1]
        self.gate_v = nn.ModuleList([nn.Linear(D * 3, D) for _ in range(4)])
        # Gating for text fusion
        self.gate_t = nn.ModuleList([nn.Linear(D * 3, D) for _ in range(4)])

        self.norm_v = nn.ModuleList([nn.LayerNorm(D) for _ in range(4)])
        self.norm_t = nn.ModuleList([nn.LayerNorm(D) for _ in range(4)])

    def forward(self, stage_outputs, text_groups, text_mask=None):
        # Build key_padding_mask for MHA (True = ignore position)
        if text_mask is not None:
            kpm = ~text_mask.bool()   # (B, seq)  True where padding
        else:
            kpm = None

        Mv, Mt = [], []
        for i in range(4):
            # ── Visual projection + spatial pooling ──────────────────────────
            v = self.v_proj[i](stage_outputs[i])   # (B, Ni, D)
            # AdaptiveAvgPool1d: (B, D, Ni) → (B, D, VISUAL_POOL)
            v_p = F.adaptive_avg_pool1d(
                v.transpose(1, 2), VISUAL_POOL
            ).transpose(1, 2)                       # (B, 196, D)

            t = text_groups[i]                      # (B, seq, D)

            # ── Cross-attention ──────────────────────────────────────────────
            v_attn, _ = self.cross_v[i](
                v_p, t, t, key_padding_mask=kpm
            )   # (B, 196, D)

            t_attn, _ = self.cross_t[i](
                t, v_p, v_p
            )   # (B, seq, D)

            # ── Gated residual ───────────────────────────────────────────────
            g_v = torch.sigmoid(
                self.gate_v[i](torch.cat([v_p, v_attn, v_p - v_attn], dim=-1))
            )
            Mv_i = self.norm_v[i](v_p + g_v * v_attn)   # (B, 196, D)

            g_t = torch.sigmoid(
                self.gate_t[i](torch.cat([t, t_attn, t - t_attn], dim=-1))
            )
            Mt_i = self.norm_t[i](t + g_t * t_attn)      # (B, seq, D)

            Mv.append(Mv_i)
            Mt.append(Mt_i)

        return Mv, Mt


class DualStateReasoning(nn.Module):
    """
    Attribute-prior driven dual-state GRU reasoning.

    Maintains two states: visual state Sv and text state St.
    Each hop:
      - Sv attends (self) to Mv_mem and (cross) to Mt_mem.
      - St attends (self) to Mt_mem and (cross) to Mv_mem.
      - Both states are updated via attribute-conditioned GRU gates.

    Args:
        Mv   : list of 4 × (B, 196, D)
        Mt   : list of 4 × (B, seq, D)
        Fa   : (B, 11, D)  attribute tokens from PARNAttributeEncoder
        text_mask : (B, seq) — 1 for real, 0 for padding (optional)

    Returns:
        (B, D*2)  — concatenation of final Sv and St
    """

    def __init__(self):
        super().__init__()

        # State initialisation projections
        self.phi_v = nn.Sequential(nn.Linear(D * 2, D), nn.Tanh())
        self.phi_t = nn.Sequential(nn.Linear(D * 2, D), nn.Tanh())

        # Self-attention heads
        self.att_v_self = nn.MultiheadAttention(D, NUM_HEADS, batch_first=True, dropout=0.1)
        self.att_t_self = nn.MultiheadAttention(D, NUM_HEADS, batch_first=True, dropout=0.1)

        # Cross-attention heads
        self.att_t2v = nn.MultiheadAttention(D, NUM_HEADS, batch_first=True, dropout=0.1)
        self.att_v2t = nn.MultiheadAttention(D, NUM_HEADS, batch_first=True, dropout=0.1)

        # Attribute prior projections
        self.W_av = nn.Linear(D, D)
        self.W_at = nn.Linear(D, D)

        # Visual GRU gates
        self.W_rv = nn.Linear(D * 4, D)   # reset  (uses attr)
        self.W_zv = nn.Linear(D * 3, D)   # update
        self.W_hv = nn.Linear(D * 3, D)   # candidate

        # Text GRU gates
        self.W_rt = nn.Linear(D * 4, D)
        self.W_zt = nn.Linear(D * 3, D)
        self.W_ht = nn.Linear(D * 3, D)

        self.hops = NUM_HOPS

    def forward(self, Mv, Mt, Fa, text_mask=None):
        # ── Build memory banks ───────────────────────────────────────────────
        Mv_mem = torch.cat(Mv, dim=1)   # (B, 4*196=784, D)
        Mt_mem = torch.cat(Mt, dim=1)   # (B, 4*seq,     D)

        # Text key-padding mask for the concatenated memory (4 repetitions)
        if text_mask is not None:
            kpm_t = ~text_mask.bool()              # (B, seq)
            kpm_t_mem = kpm_t.repeat(1, 4)         # (B, 4*seq)
        else:
            kpm_t_mem = None

        # ── Attribute prior vectors ──────────────────────────────────────────
        Fa_mean = Fa.mean(dim=1)        # (B, D)
        a_v = self.W_av(Fa_mean)        # (B, D)
        a_t = self.W_at(Fa_mean)        # (B, D)

        # ── Initialise dual states from the two deepest levels ───────────────
        Sv = self.phi_v(torch.cat([Mv[-1].mean(1), Mv[-2].mean(1)], dim=-1))  # (B, D)
        St = self.phi_t(torch.cat([Mt[-1].mean(1), Mt[-2].mean(1)], dim=-1))  # (B, D)

        # ── Iterative GRU reasoning ──────────────────────────────────────────
        for _ in range(self.hops):
            # Visual self-attention: Sv queries Mv_mem
            v_self, _ = self.att_v_self(
                Sv.unsqueeze(1), Mv_mem, Mv_mem
            )
            v_self = v_self.squeeze(1)   # (B, D)

            # Text→Visual cross-attention: Sv queries Mt_mem
            t2v, _ = self.att_t2v(
                Sv.unsqueeze(1), Mt_mem, Mt_mem,
                key_padding_mask=kpm_t_mem,
            )
            t2v = t2v.squeeze(1)         # (B, D)

            # Text self-attention: St queries Mt_mem
            t_self, _ = self.att_t_self(
                St.unsqueeze(1), Mt_mem, Mt_mem,
                key_padding_mask=kpm_t_mem,
            )
            t_self = t_self.squeeze(1)   # (B, D)

            # Visual→Text cross-attention: St queries Mv_mem
            v2t, _ = self.att_v2t(
                St.unsqueeze(1), Mv_mem, Mv_mem
            )
            v2t = v2t.squeeze(1)         # (B, D)

            # ── GRU update: visual state ─────────────────────────────────────
            r_v = torch.sigmoid(self.W_rv(torch.cat([Sv, v_self, t2v, a_v], dim=-1)))
            z_v = torch.sigmoid(self.W_zv(torch.cat([Sv, v_self, t2v],      dim=-1)))
            h_v = torch.tanh(   self.W_hv(torch.cat([r_v * Sv, v_self, t2v], dim=-1)))
            Sv = (1 - z_v) * Sv + z_v * h_v

            # ── GRU update: text state ───────────────────────────────────────
            r_t = torch.sigmoid(self.W_rt(torch.cat([St, t_self, v2t, a_t], dim=-1)))
            z_t = torch.sigmoid(self.W_zt(torch.cat([St, t_self, v2t],      dim=-1)))
            h_t = torch.tanh(   self.W_ht(torch.cat([r_t * St, t_self, v2t], dim=-1)))
            St = (1 - z_t) * St + z_t * h_t

        return torch.cat([Sv, St], dim=-1)   # (B, D*2 = 4096)


class catNet(nn.Module):
    """
    Full AADB attribute-prior hierarchical cross-modal dual-state network.

    Args:
        bert               : pre-loaded BertModel
        parn_pretrained_path : optional path to a PARN checkpoint for the
                               attribute encoder (None = random init)
        freeze_parn        : freeze PARN encoder weights after loading

    Forward:
        image      : (B, 3, 448, 448)  — ImageNet-normalised
        text       : (B, seq)          — BERT token ids
        text_mask  : (B, seq)          — 1 for real tokens, 0 for padding

    Returns:
        (B, 10) aesthetic quality distribution (Softmax)
    """

    def __init__(self, bert, parn_pretrained_path: str = None,
                 freeze_parn: bool = False):
        super().__init__()

        self.txt_enc  = EncoderText(bert)
        self.img_enc  = swin_base_patch4_window7_224_in22k()
        self.attr_enc = PARNAttributeEncoder(
            out_dim=D,
            pretrained_path=parn_pretrained_path,
            freeze=freeze_parn,
        )
        self.fusion    = HierarchicalCrossModalFusion()
        self.reasoning = DualStateReasoning()

        self.drop    = nn.Dropout(0.5)
        self.fc1     = nn.Linear(D * 2, 256)
        self.fc2     = nn.Linear(256, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image, text, text_mask):
        # ── Text: 4-level hierarchical encoding ─────────────────────────────
        text_groups = self.txt_enc(text, text_mask)   # list of 4 × (B, seq, D)

        # ── Vision: multi-stage Swin Transformer ────────────────────────────
        stage_outputs = self.img_enc(image)
        # [(B,3136,256),(B,784,512),(B,196,1024),(B,196,1024)] for 448×448

        # ── Attributes: PARN on a 224×224 resized copy ──────────────────────
        img_small = F.interpolate(
            image, size=(224, 224), mode='bilinear', align_corners=False
        )
        Fa = self.attr_enc(img_small)   # (B, 11, D)

        # ── Hierarchical cross-modal fusion ──────────────────────────────────
        Mv, Mt = self.fusion(stage_outputs, text_groups, text_mask=text_mask)

        # ── Attribute-conditioned dual-state reasoning ───────────────────────
        h = self.reasoning(Mv, Mt, Fa, text_mask=text_mask)   # (B, 4096)

        # ── Classification head ──────────────────────────────────────────────
        h = self.drop(h)
        h = F.relu(self.fc1(h))   # (B, 256)
        h = self.fc2(h)           # (B, 10)
        return self.softmax(h)
