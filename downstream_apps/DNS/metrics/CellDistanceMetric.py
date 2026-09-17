"""
Grid-distance metrics for the brightest-cell flare-location task.

CellDistanceMetric scores predictions from BrightestCellModel /
BrightestCellTrainedThresholdModel (downstream_apps/DNS/models/), which forecast the flare
location as one of 65 classes: class 0 means "no flare", classes 1..64 are cells of an 8x8
heliographic grid (row-major, 1-indexed — see downstream_apps/DNS/data/flare_grid.py).

Rather than plain classification accuracy, "how far off" a wrong prediction is matters: the
metric is the Chebyshev distance between the predicted and actual cell's (row, col) grid
position. Class 0 has no grid position, so any mismatch between "no flare" and a real cell is
scored as the grid's max distance (the diagonal); an exact 0-vs-0 match is distance 0.

Five metric sets, selected by mode at construction time (mirrors template_metrics.FlareMetrics):
- "train_loss"    — differentiable *soft* distance: the expected distance under
                    softmax(preds), so it carries gradient back to preds even where the hard,
                    argmax-based distance below does not (only the decision boundary moves
                    it). This is what trains BrightestCellTrainedThresholdModel's learned
                    no-flare threshold via trainer.fit(...).
- "val_loss"      — the quantity logged as `val_loss` and used to select checkpoints. Uses the
                    literal hard distance (argmax(preds) vs target) for an interpretable,
                    non-differentiable evaluation number.
- "train_metrics" — hard distance *and* accuracy, reported only.
- "val_metrics"   — hard distance *and* accuracy, reported only. Does NOT influence checkpoint
                    selection — "val_loss" does.
- "accuracy"      — standalone accuracy alone (see below), for callers that want just that
                    number without also computing grid_distance.

Accuracy is derived from the same hard distance as "val_loss"/"train_metrics", not from a
separate argmax(preds) == target check: a sample counts as correct iff its hard grid distance
is exactly 0. Distinct classes never share a grid position (two different real cells are at
least 1 apart; a real cell vs. "no flare" is the grid's max distance), so distance == 0 is
equivalent to an exact class match — accuracy is just that same distance computation viewed as
a hit/miss rate instead of an average magnitude.

Shape contract: preds is (B, num_classes) unnormalized scores, as produced by
BrightestCellModel.forward / BrightestCellTrainedThresholdModel.forward; target is (B,) or
(B, 1) integer class ids in [0, num_classes - 1].
"""

from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange

Mode = Literal["train_loss", "val_loss", "train_metrics", "val_metrics", "accuracy"]


def build_distance_matrix(grid_rows: int = 8, grid_cols: int = 8) -> torch.Tensor:
    """(num_classes, num_classes) pairwise Chebyshev grid distance.

    Class 0 is "no flare" and has no grid position: D[0, 0] = 0, and every other entry
    touching class 0 is the grid's max distance (the diagonal), since a no-flare/real-cell
    mismatch has no meaningful in-grid distance to fall back on. Classes 1..num_classes-1 map
    to (row, col) the same way as data/flare_grid.py's latlon_to_cell: cell = row*grid_cols +
    col + 1, row-major, 1-indexed.
    """
    num_cells = grid_rows * grid_cols
    num_classes = num_cells + 1
    max_distance = float(max(grid_rows - 1, grid_cols - 1))

    cell_idx = torch.arange(num_cells)
    row = (cell_idx // grid_cols).float()
    col = (cell_idx % grid_cols).float()
    row_diff = (rearrange(row, "n -> n 1") - rearrange(row, "n -> 1 n")).abs()
    col_diff = (rearrange(col, "n -> n 1") - rearrange(col, "n -> 1 n")).abs()
    cell_distances = torch.maximum(row_diff, col_diff)  # (num_cells, num_cells)

    distance_matrix = torch.full((num_classes, num_classes), max_distance)
    distance_matrix[0, 0] = 0.0
    distance_matrix[1:, 1:] = cell_distances
    return distance_matrix


class CellDistanceMetric:
    """Grid-distance metrics for the brightest-cell classification task. See module docstring."""

    def __init__(self, mode: Mode, grid_rows: int = 8, grid_cols: int = 8) -> None:
        """
        Args:
            mode: One of "train_loss", "val_loss", "train_metrics", "val_metrics", or
                "accuracy".
            grid_rows: Number of rows in the heliographic grid.
            grid_cols: Number of columns in the heliographic grid.
        """
        self.mode: Mode = mode
        self.grid_rows: int = grid_rows
        self.grid_cols: int = grid_cols
        self.num_classes: int = grid_rows * grid_cols + 1

        # Cache the distance matrix once (instead of rebuilding it every call).
        self._distance_matrix: torch.Tensor = build_distance_matrix(grid_rows, grid_cols)

    def _ensure_device(self, preds: torch.Tensor) -> None:
        """Move the cached distance matrix to the same device as ``preds``, if needed."""
        if self._distance_matrix.device != preds.device:
            self._distance_matrix = self._distance_matrix.to(preds.device)

    def _check_shape(self, preds: torch.Tensor) -> None:
        if preds.shape[-1] != self.num_classes:
            raise ValueError(
                f"preds has {preds.shape[-1]} classes but grid_rows={self.grid_rows}, "
                f"grid_cols={self.grid_cols} implies {self.num_classes} classes."
            )

    def _target_ids(self, target: torch.Tensor) -> torch.Tensor:
        """Flatten (B,) or (B, 1) target to (B,) class ids, regardless of dtype."""
        return rearrange(target, "... -> (...)").long()

    def _hard_distance(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-sample Chebyshev distance between argmax(preds) and target. Not differentiable."""
        self._check_shape(preds)
        self._ensure_device(preds)
        pred_ids = preds.argmax(dim=-1)
        target_ids = self._target_ids(target)
        return self._distance_matrix[pred_ids, target_ids]

    def _soft_distance(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-sample expected Chebyshev distance under softmax(preds). Differentiable."""
        self._check_shape(preds)
        self._ensure_device(preds)
        target_ids = self._target_ids(target)
        probs = F.softmax(preds, dim=-1)
        distance_to_target = rearrange(self._distance_matrix[:, target_ids], "c b -> b c")
        return (probs * distance_to_target).sum(dim=-1)

    def train_loss(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """Differentiable soft grid distance — trains BrightestCellTrainedThresholdModel."""
        return {"soft_grid_distance": self._soft_distance(preds, target).mean()}, [1.0]

    def val_loss(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """Hard grid distance — the quantity ModelCheckpoint uses to select checkpoints."""
        return {"grid_distance": self._hard_distance(preds, target).mean()}, [1.0]

    def train_metrics(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """Hard grid distance and exact-match accuracy, reported only."""
        distance = self._hard_distance(preds, target)
        return {
            "grid_distance": distance.mean(),
            "accuracy": (distance == 0).float().mean(),
        }, [1.0, 1.0]

    def val_metrics(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """Hard grid distance and exact-match accuracy, reported only. Does NOT influence
        checkpoint selection."""
        distance = self._hard_distance(preds, target)
        return {
            "grid_distance": distance.mean(),
            "accuracy": (distance == 0).float().mean(),
        }, [1.0, 1.0]

    def accuracy(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """Exact-match accuracy alone: fraction of samples with hard grid distance 0.

        train_metrics()/val_metrics() already include this under the same key, so this
        mode exists for callers that want just accuracy without also computing
        grid_distance.
        """
        distance = self._hard_distance(preds, target)
        return {"accuracy": (distance == 0).float().mean()}, [1.0]

    def __call__(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """Evaluate metrics for the mode set at construction time.

        Args:
            preds: (B, num_classes) unnormalized class scores.
            target: (B,) or (B, 1) ground-truth class ids.

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - Metric dictionary. Keys become logger metric names; values are scalar
                  tensors aggregated over the batch.
                - List of per-metric weights (used by FlareLightningModule to combine
                  multiple loss terms into a single scalar).
        """
        match self.mode.lower():

            case "train_loss":
                return self.train_loss(preds, target)

            # No torch.no_grad() here, matching "train_loss": Lightning already disables
            # gradients during validation, so wrapping it would differ gratuitously from
            # the loss case this mirrors.
            case "val_loss":
                return self.val_loss(preds, target)

            case "train_metrics":
                with torch.no_grad():
                    return self.train_metrics(preds, target)

            case "val_metrics":
                with torch.no_grad():
                    return self.val_metrics(preds, target)

            case "accuracy":
                with torch.no_grad():
                    return self.accuracy(preds, target)

            case _:
                raise NotImplementedError(
                    f"{self.mode} is not implemented as a valid metric case."
                )
