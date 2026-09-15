"""
Brightest-cell baseline: the next flare is forecast in the grid cell that is brightest now.

Nothing is trained. Score it with ``trainer.validate(...)``, never ``trainer.fit(...)``.
Use the validation DataLoader: the training one applies random vertical flips.

Grid: the 8x8 heliographic grid from ``downstream_apps/DNS/data/flare_grid.py``, cells 1..64.
Cell 0 means "no flare", so there are 65 classes.

Assumed geometry (check once by overlaying the cell map on a plotted frame):
  * disk centred in the 4096x4096 frame, solar radius ~1600 px (AIA plate scale)
  * array row 0 is solar north; pass north_up=False if predictions come out mirrored
  * the seasonal tilt of the solar axis is ignored (fine for 22.5 deg cells)
"""

import numpy as np
import torch
import torch.nn as nn

from downstream_apps.DNS.data.flare_grid import STEP  # 22.5 deg cell size

NUM_CLASSES = 65  # cells 1..64 + cell 0 = "no flare"
OFF_DISK = -1


def build_pixel_cell_map(
    size: int = 4096, radius_px: float = 1600.0, north_up: bool = True
) -> np.ndarray:
    """Return a ``(size, size)`` array with the grid cell (1..64) of every pixel.

    Off-disk pixels get ``OFF_DISK``. Built once and reused for every image.
    """
    c = (size - 1) / 2.0
    rows, cols = np.indices((size, size), dtype=np.float64)

    # Position in units of the solar radius: x toward solar west, y toward solar north.
    x = (cols - c) / radius_px
    y = (c - rows) / radius_px if north_up else (rows - c) / radius_px
    on_disk = (x * x + y * y) < 1.0

    # Sphere seen face-on: y = sin(lat), x = cos(lat) * sin(lon).
    lat = np.degrees(np.arcsin(np.clip(y, -1.0, 1.0)))
    cos_lat = np.cos(np.radians(lat))
    with np.errstate(divide="ignore", invalid="ignore"):
        lon = np.degrees(np.arcsin(np.clip(x / np.where(cos_lat > 0, cos_lat, 1.0), -1.0, 1.0)))

    # Same arithmetic as flare_grid.latlon_to_cell, applied to the whole image at once.
    col = np.clip(np.floor((lon + 90.0) / STEP), 0, 7)   # 0 at the east limb
    row = np.clip(np.floor((90.0 - lat) / STEP), 0, 7)   # 0 at the north pole
    cell_map = (row * 8 + col + 1).astype(np.int64)
    cell_map[~on_disk] = OFF_DISK
    return cell_map


class BrightestCellModel(nn.Module):
    """Mean brightness of one channel per cell; the brightest cell is the prediction.

    Args:
        channel_order: ``cfg.data.channels``, the order of the C axis of ``batch["ts"]``.
        channel: Channel to measure brightness in (hot coronal channels track active regions).
        no_flare_threshold: Predict cell 0 when the brightest cell is below this value.
            Units are the dataset's normalised space, not physical. ``None`` never predicts 0.
        cell_map: From ``build_pixel_cell_map``; built with defaults if omitted.

    Input: batch dict with ``ts`` of shape ``(B, C, T, H, W)``.
    Output: ``(B, 65)`` scores, so ``argmax(dim=1)`` is the predicted cell and the same
    metrics work for this baseline and for a trained classifier.
    """

    def __init__(
        self,
        channel_order: list[str],
        channel: str = "aia94",
        no_flare_threshold: float | None = None,
        cell_map: np.ndarray | None = None,
    ):
        super().__init__()
        self.channel_idx = list(channel_order).index(channel)
        self.no_flare_threshold = no_flare_threshold

        cell_map = torch.as_tensor(build_pixel_cell_map() if cell_map is None else cell_map)
        on_disk = cell_map.flatten() >= 1
        # Flat positions of the on-disk pixels and the cell each one belongs to.
        # Buffers follow .to(device) but are never optimised.
        self.register_buffer("on_disk_idx", torch.nonzero(on_disk).squeeze(1))
        self.register_buffer("on_disk_cell", cell_map.flatten()[on_disk])
        counts = torch.bincount(self.on_disk_cell, minlength=NUM_CLASSES).clamp(min=1)
        self.register_buffer("cell_counts", counts.float())

    @torch.no_grad()
    def forward(self, batch: dict) -> torch.Tensor:
        x = batch["ts"][:, self.channel_idx, 0]                # (B, H, W), first timestep
        pixels = x.reshape(x.shape[0], -1)[:, self.on_disk_idx]  # (B, n_on_disk)

        # index_add_ pours every pixel into the bucket of its cell; divide to get means.
        sums = torch.zeros(x.shape[0], NUM_CLASSES, dtype=pixels.dtype, device=pixels.device)
        sums.index_add_(1, self.on_disk_cell, pixels)
        scores = sums / self.cell_counts.to(pixels.dtype)

        # Column 0 is "no flare": it wins only if every cell is below the threshold.
        scores[:, 0] = (
            torch.finfo(scores.dtype).min if self.no_flare_threshold is None else self.no_flare_threshold
        )
        return scores
