#!/usr/bin/env python
# coding: utf-8
"""
Validation-only script based on 1_BrightestCellBaseline.ipynb.

BrightestCellModel has no trainable parameters, so there is nothing to fit: this script
runs it through trainer.validate() on the validation split only (per BrightestCellModel's
own docstring) and prints the resulting grid_distance / accuracy. Contrast with
1_BrightestCellTrainedBaseline.py, which trains BrightestCellTrainedThresholdModel's
learned no-flare threshold via trainer.fit().

The dataloaders use multiprocessing_context="spawn" (workshop_infrastructure/datasets/
builders.py): the dataset holds an S3 client that does not survive fork. Under spawn,
every worker process re-imports this file as a module to get its code. All executable
logic therefore lives inside main(), guarded by `if __name__ == "__main__":` below --
without that guard, each worker would re-run the whole script (re-loading the config,
rebuilding the dataloaders, and spawning its own workers) instead of just importing
definitions.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Must be set BEFORE torch is imported: cuBLAS reads this once, when it initializes, so
# setting it later has no effect.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys

import torch

import lightning as L
from lightning.pytorch.loggers import CSVLogger

# Append base path so the script can see the workshop_infrastructure folder.
sys.path.append("../../")

from workshop_infrastructure.utils import build_scalers  # Data scaling utilities for Surya stacks
from downstream_apps.DNS.configs import load_flare_config
from workshop_infrastructure.assets import ensure_assets
from downstream_apps.DNS.datasets.Hinode_Flare_Dataset import HinodeFlareDataset
from workshop_infrastructure.datasets.builders import build_helio_dataloaders
from downstream_apps.DNS.models.brightest_cell_baseline import BrightestCellModel
from downstream_apps.DNS.metrics.CellDistanceMetric import CellDistanceMetric
from downstream_apps.template.lightning_modules.pl_simple_baseline import FlareLightningModule


def main() -> None:
    torch.set_float32_matmul_precision('medium')

    # -- Load configuration --
    # baseline_untrained.yaml is a fork of config_script.yaml that restricts data.channels
    # to just aia304 -- the only channel BrightestCellModel reads (see the model
    # construction below) -- and leaves data.max_samples at null, so the reported
    # accuracy covers the whole validation split by default.
    cfg = load_flare_config("./configs/baseline_untrained.yaml")
    print(f"Loaded config for job: {cfg.job_id}")

    # -- Download assets --
    # This baseline has no backbone, so skip the 1.8 GB pretrained weights.
    ensure_assets(cfg, which=["scalers"])
    scalers = build_scalers(info=cfg.data.scalers_path)
    print(f"Loaded scalers for {len(scalers)} channels.")

    # -- Define Downstream (DS) datasets --
    # build_helio_dataloaders() always builds both splits; only val_data_loader is used
    # below, since BrightestCellModel is untrained and the training loader applies random
    # vertical flips that would make the "accuracy" number depend on an augmentation this
    # model was never fit to.
    _, val_data_loader = build_helio_dataloaders(
        cfg,
        HinodeFlareDataset,
        scalers=scalers,
        num_workers=cfg.num_workers,
        #### Downstream (DS) specific parameters
        return_surya_stack=True,
        max_number_of_samples=cfg.data.max_samples,
        ds_flare_index_path=cfg.data.flare_index_path,
        ds_time_column=cfg.data.ds_time_column,
        ds_time_tolerance=cfg.data.ds_time_tolerance,
        ds_match_direction=cfg.data.ds_match_direction,
    )

    print(f"val: {len(val_data_loader.dataset)} samples | batch_size: {cfg.batch_size}")

    # Inspect a single batch to confirm shapes before validating.
    batch = next(iter(val_data_loader))
    print({k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__) for k, v in batch.items()})

    # -- Define the untrained baseline model --
    # channel_order comes straight from cfg.data.channels (just ["aia304"] in
    # baseline_untrained.yaml), so batch["ts"] arrives with 1 channel, not 13, and
    # channel_idx below resolves to 0. Nothing here is trained: forward() is decorated
    # with @torch.no_grad() and every attribute is a buffer, never a parameter.
    model = BrightestCellModel(cfg.data.channels, channel="aia304")

    scores = model(batch)
    predicted = scores.argmax(dim=1)
    print("predicted cell:", predicted.tolist())
    print("true cell:", batch["forecast"].tolist())

    # -- Define your metrics --
    # FlareLightningModule requires train_loss/train_metrics keys even though this script
    # never calls trainer.fit(); they are wired for shape compatibility only and never
    # backpropagate anything, since BrightestCellModel has no parameters to update.
    train_loss_metrics = CellDistanceMetric("train_loss")
    # val_loss is the quantity logged as "val_loss"; grid_distance is what it reports here.
    val_loss_metrics = CellDistanceMetric("val_loss")
    train_evaluation_metrics = CellDistanceMetric("train_metrics")
    # val_metrics reports both grid_distance and accuracy -- this is what answers "how
    # accurate is this model", logged automatically as val_metric_accuracy below.
    validation_evaluation_metrics = CellDistanceMetric("val_metrics")
    # Standalone accuracy mode, exercised here on its own to confirm it works in isolation.
    accuracy_metrics = CellDistanceMetric("accuracy")

    accuracy_dict, _ = accuracy_metrics(scores, batch["forecast"])
    print("accuracy on preview batch:", accuracy_dict["accuracy"].item())

    # -- Define your PyTorch lightning module --
    metrics = {
        'train_loss': train_loss_metrics,
        'val_loss': val_loss_metrics,
        'train_metrics': train_evaluation_metrics,
        'val_metrics': validation_evaluation_metrics,
    }

    lit_model = FlareLightningModule(
        model, metrics, lr=cfg.learning_rate, batch_size=cfg.batch_size
    )

    # -- Initialize trainer --
    # No ModelCheckpoint / WandB: nothing trains, so there is no checkpoint to select and
    # no training curve worth tracking. CSVLogger keeps a record of the run on disk.
    csv_logger = CSVLogger("runs", name=cfg.wandb_project)

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        logger=csv_logger,
        log_every_n_steps=2,
    )

    # -- Validate the model --
    # trainer.validate(), not trainer.fit(): per BrightestCellModel's own docstring,
    # nothing here is trained, so there is nothing to fit.
    results = trainer.validate(lit_model, dataloaders=val_data_loader)
    print("validation results:", results)


if __name__ == "__main__":
    main()
