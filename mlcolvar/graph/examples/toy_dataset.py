"""Utility functions to build a toy graph dataset for quick experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from mlcolvar.graph.data import atomic
from mlcolvar.graph.data.dataset import GraphDataSet, create_dataset_from_configurations


@dataclass
class ToyDatasetConfig:
    """Configuration parameters for the toy dataset generator.

    Attributes
    ----------
    n_samples:
        Number of molecular configurations to generate.
    cutoff:
        Graph cutoff radius used when building edges.
    noise:
        Standard deviation of the Gaussian noise added to atomic positions.
    random_seed:
        Seed for the random number generator. ``None`` disables seeding.
    save_path:
        Optional path where the dataset will be serialized with
        :func:`mlcolvar.graph.data.dataset.save_dataset`.
    """

    n_samples: int = 32
    cutoff: float = 4.5
    noise: float = 0.05
    random_seed: int | None = 0
    save_path: Path | None = None


def _water_like_geometry(state: int) -> np.ndarray:
    """Return a simple three-atom geometry mimicking a water molecule.

    Two metastable states are produced by displacing the hydrogen atoms in
    opposite directions. The resulting coordinates are expressed in Ångström.
    """

    oxygen = np.array([0.0, 0.0, 0.0])
    hydrogens = np.array([
        [0.9572, 0.0, 0.0],
        [-0.2390, 0.9266, 0.0],
    ])

    if state == 0:
        displacement = np.array([0.0, -0.05, 0.0])
    else:
        displacement = np.array([0.0, 0.05, 0.0])

    positions = np.vstack([oxygen, hydrogens + displacement])
    return positions


def generate_configurations(
    n_samples: int,
    noise: float,
    random_seed: int | None = None,
) -> Sequence[atomic.Configuration]:
    """Create a list of :class:`~mlcolvar.graph.data.atomic.Configuration` objects."""

    rng = np.random.default_rng(random_seed)

    atomic_numbers = np.array([8, 1, 1])
    cell = np.eye(3) * 10.0
    pbc = (False, False, False)

    configurations = []
    for i in range(n_samples):
        state = i % 2
        base_positions = _water_like_geometry(state)
        noisy_positions = base_positions + rng.normal(scale=noise, size=base_positions.shape)
        configurations.append(
            atomic.Configuration(
                atomic_numbers=atomic_numbers,
                positions=noisy_positions,
                cell=cell,
                pbc=pbc,
                node_labels=None,
                graph_labels=np.array([[state]]),
            )
        )

    return configurations


def build_toy_dataset(config: ToyDatasetConfig | None = None) -> GraphDataSet:
    """Generate and (optionally) persist a small :class:`GraphDataSet`.

    Parameters
    ----------
    config:
        Optional dataclass collecting all generation parameters. When ``None``
        the default :class:`ToyDatasetConfig` settings are used.
    """

    config = config or ToyDatasetConfig()

    configurations = generate_configurations(
        n_samples=config.n_samples,
        noise=config.noise,
        random_seed=config.random_seed,
    )

    z_table = atomic.AtomicNumberTable.from_zs(configurations[0].atomic_numbers)
    dataset = create_dataset_from_configurations(
        configurations,
        z_table=z_table,
        cutoff=config.cutoff,
        show_progress=False,
    )

    if config.save_path is not None:
        from mlcolvar.graph.data.dataset import save_dataset

        config.save_path.parent.mkdir(parents=True, exist_ok=True)
        save_dataset(dataset, str(config.save_path))

    return dataset


def main() -> None:
    """Build the toy dataset and report its basic properties."""

    dataset = build_toy_dataset()
    print(dataset)
    sample = dataset[0]
    print("Number of nodes:", sample.num_nodes)
    print("Graph label:", sample.graph_labels.squeeze().item())


if __name__ == "__main__":
    main()
