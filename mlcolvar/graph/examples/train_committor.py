"""Train a GraphCommittor collective variable on the toy dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import torch

from mlcolvar.graph.cvs.committor.committor import GraphCommittor
from mlcolvar.graph.data import atomic
from mlcolvar.graph.data.datamodule import GraphDataModule
from mlcolvar.graph.examples.toy_dataset import ToyDatasetConfig, build_toy_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./graph_committor_demo"),
        help="Directory where logs and checkpoints will be written.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs to run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Mini-batch size used by the dataloader.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_toy_dataset(
        ToyDatasetConfig(n_samples=128, cutoff=4.5, noise=0.08)
    )
    for graph in dataset:
        graph.weight = torch.ones(1, dtype=torch.get_default_dtype())

    datamodule = GraphDataModule(
        dataset,
        lengths=(0.7, 0.3),
        batch_size=args.batch_size,
        random_split=True,
        seed=0,
    )

    atomic_masses = atomic.get_masses(dataset.atomic_numbers)
    committor = GraphCommittor(
        cutoff=dataset.cutoff,
        atomic_numbers=dataset.atomic_numbers,
        atomic_masses=atomic_masses,
        model_options={
            "n_layers": 2,
            "n_messages": 2,
            "n_feedforwards": 1,
            "n_scalars_node": 16,
            "n_vectors_node": 8,
            "n_scalars_edge": 16,
            "drop_rate": 0.1,
        },
        optimizer_options={
            "name": "adam",
            "lr": 5e-4,
        },
    )

    logger = CSVLogger(save_dir=str(output_dir), name="committor")
    callbacks = [
        EarlyStopping(monitor="valid_loss", patience=5, mode="min"),
    ]

    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=args.epochs,
        logger=logger,
        default_root_dir=str(output_dir),
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    trainer.fit(committor, datamodule=datamodule)
    checkpoint_path = output_dir / "graph_committor.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
