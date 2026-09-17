#!/usr/bin/env python
# coding: utf-8
"""
Converted from 1_BrightestCellTrainedBaseline.ipynb.

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
# setting it later has no effect. It is what lets training.deterministic work without a
# cuBLAS warning on every run.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
from torch.utils.data import DataLoader

import torch
import yaml

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

# Append base path so the script can see the workshop_infrastructure folder.
sys.path.append("../../")

from workshop_infrastructure.utils import build_scalers  # Data scaling utilities for Surya stacks
from downstream_apps.DNS.configs import load_flare_config
from workshop_infrastructure.assets import ensure_assets
from downstream_apps.DNS.datasets.Hinode_Flare_Dataset import HinodeFlareDataset
from workshop_infrastructure.datasets.builders import build_helio_dataloaders
from downstream_apps.DNS.models.brightest_cell_trained_threshold import BrightestCellTrainedThresholdModel
from downstream_apps.DNS.metrics.CellDistanceMetric import CellDistanceMetric
from downstream_apps.template.lightning_modules.pl_simple_baseline import FlareLightningModule


def main() -> None:
    torch.set_float32_matmul_precision('medium')

    # -- Load configuration --
    # baseline_threshold.yaml is a fork of config_script.yaml that restricts data.channels
    # to just aia304 -- the only channel BrightestCellTrainedThresholdModel reads (see the
    # model construction below). Loading all 13 Surya channels here would decode and
    # normalize 12 the model never touches, for no benefit: this baseline has no backbone,
    # so nothing else in this script needs the other channels.
    cfg = load_flare_config("./configs/baseline_threshold.yaml")
    print(f"Loaded config for job: {cfg.job_id}")

    # -- Download assets --
    # The linear baseline needs no backbone, so skip the 1.8 GB weights.
    ensure_assets(cfg, which=["scalers"])
    scalers = build_scalers(info=cfg.data.scalers_path)
    print(f"Loaded scalers for {len(scalers)} channels.")

    # -- Define Downstream (DS) datasets --
    # build_helio_dataloaders() constructs the train and validation datasets and wraps them
    # in DataLoaders. Only the flare-specific arguments are passed here.
    train_data_loader, val_data_loader = build_helio_dataloaders(
        cfg,
        HinodeFlareDataset,
        scalers=scalers,
        num_workers=4,
        #### Downstream (DS) specific parameters
        return_surya_stack=True,
        max_number_of_samples=1000,
        ds_flare_index_path=cfg.data.flare_index_path,
        ds_time_column=cfg.data.ds_time_column,
        ds_time_tolerance=cfg.data.ds_time_tolerance,
        ds_match_direction=cfg.data.ds_match_direction,
    )

    batch_size = cfg.batch_size
    print(f"train: {len(train_data_loader.dataset)} samples | "
          f"val: {len(val_data_loader.dataset)} samples | batch_size: {batch_size}")

    # Inspect a single batch to confirm shapes before training.
    batch = next(iter(train_data_loader))
    print({k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__) for k, v in batch.items()})

    # -- Define simple baseline model --
    # channel_order comes straight from cfg.data.channels (now just ["aia304"] in
    # baseline_threshold.yaml), so batch["ts"] arrives with 1 channel, not 13, and
    # channel_idx below resolves to 0.
    model = BrightestCellTrainedThresholdModel(cfg.data.channels, channel="aia304")

    scores = model(batch)
    predicted = scores.argmax(dim=1)
    print("predicted cell:", predicted.tolist())
    print("true cell:", batch["forecast"].tolist())

    # -- Define your metrics --
    train_loss_metrics = CellDistanceMetric("train_loss")
    # val_loss is the quantity logged as "val_loss" and used to pick the best checkpoint.
    val_loss_metrics = CellDistanceMetric("val_loss")
    # train_metrics/val_metrics each report grid_distance AND accuracy (exact-match rate),
    # so accuracy is already logged automatically as train_metric_accuracy/val_metric_accuracy
    # during trainer.fit(...) below -- no extra wiring into `metrics` needed for that.
    train_evaluation_metrics = CellDistanceMetric("train_metrics")
    # Reported only: val_metrics do NOT influence checkpoint selection.
    validation_evaluation_metrics = CellDistanceMetric("val_metrics")
    # Standalone accuracy mode, exercised here on its own to confirm it works in isolation.
    accuracy_metrics = CellDistanceMetric("accuracy")

    train_loss_metrics(scores, batch["forecast"])
    train_evaluation_metrics(scores, batch["forecast"])
    validation_evaluation_metrics(scores, batch["forecast"])
    accuracy_dict, _ = accuracy_metrics(scores, batch["forecast"])
    print("accuracy:", accuracy_dict["accuracy"].item())

    # -- Define your PyTorch lightning module --
    L.seed_everything(42, workers=True)

    metrics = {
        'train_loss': train_loss_metrics,
        'val_loss': val_loss_metrics,
        'train_metrics': train_evaluation_metrics,
        'val_metrics': validation_evaluation_metrics,
    }

    lit_model = FlareLightningModule(
        model, metrics, lr=cfg.learning_rate, batch_size=batch_size
    )

    # -- Logging --
    project_name = cfg.wandb_project
    run_name = "baseline_experiment_1"  # give your run a descriptive name

    wandb_logger = WandbLogger(
        entity=cfg.wandb_entity,  # set wandb_entity in the config; null = personal account
        project=project_name,
        name=run_name,
        log_model=False,
        save_dir="./wandb/wandb_tmp",
    )

    csv_logger = CSVLogger("runs", name=project_name)

    # -- Initialize trainer --
    max_epochs = 2

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        logger=[wandb_logger, csv_logger],
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            )
        ],
        log_every_n_steps=2,
    )

    # -- Fit the model --
    print("threshold before training:", model.threshold.item())
    trainer.fit(lit_model, train_data_loader, val_data_loader)
    print("threshold after training:", model.threshold.item())


if __name__ == "__main__":
    main()
