"""Compute Kolmogorov bias values from a trained GraphCommittor model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mlcolvar.graph.cvs.committor import GraphCommittor
from mlcolvar.graph.cvs.committor.utils import (
    compute_committor_weights,
    get_dataset_kolmogorov_bias,
)
from mlcolvar.graph.data import atomic
from mlcolvar.graph.data.dataset import save_dataset
from mlcolvar.graph.examples.toy_dataset import ToyDatasetConfig, build_toy_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Checkpoint produced by train_committor.py.",
    )
    parser.add_argument(
        "--output-bias",
        type=Path,
        default=None,
        help="Optional path where the bias values will be stored as a .npy file.",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=None,
        help=(
            "Optional path where the dataset with updated committor weights will "
            "be serialized using save_dataset."
        ),
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Inverse temperature beta used for the bias and weights.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Regularisation constant added inside the logarithm.",
    )
    parser.add_argument(
        "--lambd",
        type=float,
        default=1.0,
        help="Global scaling factor applied to the bias values.",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Scale the bias with atomic masses when accumulating gradients.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Mini-batch size used when evaluating the committor model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = build_toy_dataset(
        ToyDatasetConfig(n_samples=128, cutoff=4.5, noise=0.08)
    )
    for graph in dataset:
        graph.weight = torch.ones(1, dtype=torch.get_default_dtype())

    atomic_masses = atomic.get_masses(dataset.atomic_numbers)
    model = GraphCommittor.load_from_checkpoint(
        args.checkpoint,
        cutoff=dataset.cutoff,
        atomic_numbers=dataset.atomic_numbers,
        atomic_masses=atomic_masses,
        map_location="cpu",
    )
    model.eval()

    bias = get_dataset_kolmogorov_bias(
        model,
        dataset,
        beta=args.beta,
        epsilon=args.epsilon,
        lambd=args.lambd,
        weighted=args.weighted,
        batch_size=args.batch_size,
    )
    print(f"Computed Kolmogorov bias statistics: mean={bias.mean():.4f}, std={bias.std():.4f}")

    updated_dataset = compute_committor_weights(dataset, bias, beta=args.beta)

    if args.output_bias is not None:
        args.output_bias.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_bias, bias)
        print(f"Saved bias values to {args.output_bias.resolve()}")

    if args.output_dataset is not None:
        args.output_dataset.parent.mkdir(parents=True, exist_ok=True)
        save_dataset(updated_dataset, str(args.output_dataset))
        print(f"Stored weighted dataset at {args.output_dataset.resolve()}")


if __name__ == "__main__":
    main()
