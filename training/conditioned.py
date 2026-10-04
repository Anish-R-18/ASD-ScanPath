"""
Phase 2: ASD/TD-conditioned TPP-Gaze.

Smallest clean modification that keeps the ORIGINAL TPP-Gaze architecture and
pretrained weights intact:

  * ONE shared base model (the unmodified official TPPGaze).
  * ONE small learnable group embedding  nn.Embedding(2, context_dim)
        ASD -> 0 , TD -> 1
  * The embedding is ADDED to the fused context (= [history || image_metadata])
    broadcast over the sequence, immediately before the duration/position heads
    (done inside model_utils.tppgaze_nll via its `cond` argument).

The embedding is ZERO-initialised, so at initialisation the conditioned model is
numerically identical to Phase 1 and the Phase-1 checkpoint loads unchanged into
`self.base` (strict). Training then learns an additive per-group shift of the
representation feeding both heads. The original tppgaze/ code is not edited.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from training import model_utils as mu


class ConditionedTPPGaze(nn.Module):
    def __init__(self, base: nn.Module, cond_dim: int, n_groups: int = 2):
        super().__init__()
        self.base = base                       # unmodified official TPPGaze
        self.cond_dim = cond_dim
        self.n_groups = n_groups
        self.group_emb = nn.Embedding(n_groups, cond_dim)
        nn.init.zeros_(self.group_emb.weight)  # identity at init == Phase 1

    def condition(self, group_ids: torch.Tensor) -> torch.Tensor:
        """(B,) group ids -> (B, cond_dim) additive condition vector."""
        return self.group_emb(group_ids)

    def nll(self, inter_times, marks, mask, metadata, group_ids):
        cond = self.condition(group_ids)
        return mu.tppgaze_nll(self.base, inter_times, marks, mask, metadata, cond=cond)
