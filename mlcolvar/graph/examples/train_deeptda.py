"""Train a GraphDeepTDA collective variable on the toy dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from mlcolvar.graph.cvs.supervised.deeptda import GraphDeepTDA
from mlcolvar.graph.data.datamodule import GraphDataModule
from mlcolvar.graph.examples.toy_dataset import ToyDatasetConfig, build_toy_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./graph_cvs_demo"),
        help="Directory where logs and checkpoints will be written.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_toy_dataset(ToyDatasetConfig(n_samples=64, cutoff=4.5, noise=0.1))
    datamodule = GraphDataModule(
        dataset,
        lengths=(0.8, 0.2),
        batch_size=16,
        random_split=True,
        seed=0,
    )

    cv = GraphDeepTDA(
        n_cvs=1,
        cutoff=dataset.cutoff,
        atomic_numbers=dataset.atomic_numbers,
        target_centers=[[-1.0], [1.0]],
        target_sigmas=[[0.4], [0.4]],
        model_options={
            "n_layers": 2,
            "n_messages": 2,
            "n_feedforwards": 1,
            "n_scalars_node": 16,
            "n_vectors_node": 8,
            "n_scalars_edge": 16,
            "drop_rate": 0.1,
        },
    )

    logger = CSVLogger(save_dir=str(output_dir), name="deeptda")
    callbacks = [
        EarlyStopping(monitor="valid_loss", patience=3, mode="min"),
    ]

    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=args.epochs,
        logger=logger,
        default_root_dir=str(output_dir),
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    trainer.fit(cv, datamodule=datamodule)
    checkpoint_path = output_dir / "deeptda.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
