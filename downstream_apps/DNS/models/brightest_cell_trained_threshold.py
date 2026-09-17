"""
Brightest-cell baseline with a trained "no flare" threshold.

Same brightest-cell scoring as ``brightest_cell_baseline.BrightestCellModel``, but the
column-0 ("no flare") score is an ``nn.Parameter`` instead of a fixed constant, so it is
learned by ``trainer.fit(...)``. Every other value is still just a per-cell brightness
mean with no learnable weights, so this isolates the effect of choosing the no-flare
cutoff optimally rather than by hand.

Use the validation DataLoader for evaluation exactly as with the untrained baseline: the
training one applies random vertical flips. See ``brightest_cell_baseline.py`` for the
grid layout, class numbering, and the geometry assumptions behind ``build_pixel_cell_map``
(reused here rather than redefined).
"""

import torch
import torch.nn as nn

from downstream_apps.DNS.models.brightest_cell_baseline import NUM_CLASSES, build_pixel_cell_map


class BrightestCellTrainedThresholdModel(nn.Module):
    """Mean brightness of one channel per cell; a learned scalar sets the "no flare" score.

    Args:
        channel_order: ``cfg.data.channels``, the order of the C axis of ``batch["ts"]``.
        channel: Channel to measure brightness in (hot coronal channels track active regions).
        initial_threshold: Starting value for the learned "no flare" score (column 0),
            in the dataset's normalised space. Updated by ``trainer.fit(...)``.
        cell_map: From ``build_pixel_cell_map``; built with defaults if omitted.

    Input: batch dict with ``ts`` of shape ``(B, C, T, H, W)``.
    Output: ``(B, 65)`` scores, so ``argmax(dim=1)`` is the predicted cell and the same
    loss/metrics work here as for the untrained baseline and a trained classifier.
    """

    def __init__(
        self,
        channel_order: list[str],
        channel: str = "aia94",
        initial_threshold: float = 0.0,
        cell_map: torch.Tensor | None = None,
    ):
        super().__init__()
        self.channel_idx = list(channel_order).index(channel)
        self.threshold = nn.Parameter(torch.tensor(float(initial_threshold)))

        cell_map = torch.as_tensor(build_pixel_cell_map() if cell_map is None else cell_map)
        on_disk = cell_map.flatten() >= 1
        # Flat positions of the on-disk pixels and the cell each one belongs to.
        # Buffers follow .to(device) but are never optimised (unlike self.threshold).
        self.register_buffer("on_disk_idx", torch.nonzero(on_disk).squeeze(1))
        self.register_buffer("on_disk_cell", cell_map.flatten()[on_disk])
        counts = torch.bincount(self.on_disk_cell, minlength=NUM_CLASSES).clamp(min=1)
        self.register_buffer("cell_counts", counts.float())

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch["ts"][:, self.channel_idx, 0]                  # (B, H, W), first timestep
        pixels = x.reshape(x.shape[0], -1)[:, self.on_disk_idx]  # (B, n_on_disk)

        # index_add_ pours every pixel into the bucket of its cell; divide to get means.
        sums = torch.zeros(x.shape[0], NUM_CLASSES, dtype=pixels.dtype, device=pixels.device)
        sums.index_add_(1, self.on_disk_cell, pixels)
        scores = sums / self.cell_counts.to(pixels.dtype)

        # Column 0 is "no flare": a learned scalar, broadcast over the batch. No
        # torch.no_grad() here (unlike the untrained baseline) — gradients must reach
        # self.threshold for trainer.fit(...) to update it.
        scores[:, 0] = self.threshold
        return scores
