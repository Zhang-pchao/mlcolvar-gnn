# The Graph Neural Network (GNN) Module in MLCOLVAR.

<p align="center">
    <img src=".github/logo.png" width="400"/>
</p>

## INTRODUCTION

The `mlcolvar.graph` package gathers all graph neural network (GNN) utilities
needed to learn collective variables (CVs) directly from atomic simulations.
It provides data processing pipelines that transform molecular configurations
into periodic neighbor graphs, several equivariant neural architectures
tailored to atomistic learning, Lightning-compatible training loops, and
post-hoc explainability tools. Together these components enable end-to-end
training of differentiable CVs that map configurations to low-dimensional
representations suitable for enhanced sampling or kinetics analyses.

## PACKAGE LAYOUT

The subpackages in this directory are organised as follows:

| Subpackage | Purpose |
|------------|---------|
| [`core`](core) | Implementations of the available GNN architectures together with their shared building blocks (radial embeddings, message passing layers, etc.). |
| [`data`](data) | Helper classes to convert atomistic configurations or trajectories into `torch_geometric` graphs, manage datasets, and interface with PyTorch Lightning data modules. |
| [`cvs`](cvs) | Lightning modules that wrap a chosen GNN and expose a common interface for training CVs. |
| [`explain`](explain) | Sensitivity analysis utilities to interpret trained CVs. |
| [`utils`](utils) | General-purpose helpers for graph construction, I/O, batching, progress reporting, and time-lagged dataset generation. |

The top-level [`__init__.py`](__init__.py) wires these components together,
initialising numerical defaults and handling optional dependencies that are
problematic to compile on some platforms.

## CORE GNN MODELS

`mlcolvar.graph.core` exposes three neural architectures via a unified
`BaseModel` interface: a Geometric Vector Perceptron (GVP), PaiNN, and SchNet
variant.
`BaseModel` prepares rotationally aware edge features by combining distance
vectors with learnable radial basis expansions.
Each specialised model then defines its message passing blocks, attention
mechanisms, and readout strategy while supporting scatter-based aggregation
over variable-size molecular graphs.
Supplementary modules contain implementations of radial basis functions,
equivariant GVP layers, and PaiNN building blocks that the models reuse.

## DATA PIPELINE

The `data` subpackage converts raw molecular information into graph objects
consumable by the models. `GraphDataSet` is a thin list-like wrapper that
stores `torch_geometric.data.Data` graphs together with metadata such as the
atomic number table and cutoffs.
Helper functions construct these graphs from individual configurations by
finding neighbour lists (with optional subsystem/environment handling),
encoding species as one-hot vectors, and attaching periodic boundary shifts
and labels.

`GraphDataModule` offers a PyTorch Lightning data module that automatically
splits a dataset into train/validation/test subsets, builds batched data
loaders, and keeps track of the split sizes for reporting.
Additional utilities in `mlcolvar.graph.utils.io` load trajectories with
MDTraj, optionally filter atoms by selection strings, and assemble graph
datasets in parallel.

## COLLECTIVE VARIABLE TRAINING

`GraphBaseCV` is a Lightning module that couples any of the supported GNNs to a
training loop for learning CVs.
It manages hyper-parameters, builds the requested model, exposes a standard
`forward` pass, and configures optimisers and schedulers while recording
metadata such as the training timestamp.
Downstream CV implementations inherit from this base class and specialise the
loss definition or logging behaviour as needed.

## EXPLAINABILITY TOOLS

The `explain` package currently provides sensitivity analysis routines that
measure how each atomic position influences a trained CV. Gradients are
accumulated over the dataset and aggregated into per-atom sensitivity scores,
helping identify the structural features most relevant to the learned order
parameters.

## UTILITIES

General utilities provide support functionality required across the module:

* `torch_tools` re-implements common `torch_scatter` operations, enforces a
  global floating-point dtype, and offers helpers to compute edge vectors and
  one-hot encodings without relying on optional compiled extensions.
* `timelagged` creates paired datasets separated by a user-defined lag time,
  optionally accounting for biased simulations through time rescaling or
  weights, which is crucial for time-lagged machine learning approaches such
  as DeepTICA.
* `progress` wraps progress-bar utilities, and other helper modules centralise
  reusable functionality shared by the rest of the package.

## TYPICAL WORKFLOW

1. **Build a dataset:** Load trajectories or configurations into a
   `GraphDataSet` using `create_dataset_from_configurations` or
   `create_dataset_from_trajectories`.
2. **Prepare loaders:** Wrap the dataset in a `GraphDataModule` (or a
   combined loader) to obtain Lightning-compatible data loaders.
3. **Instantiate a CV:** Create a `GraphBaseCV` subclass with the desired GNN
   architecture and training hyper-parameters.
4. **Train and evaluate:** Use PyTorch Lightning trainers to optimise the CV,
   and run sensitivity analysis from `mlcolvar.graph.explain` to interpret the
   learned representation.

## QUICKSTART SCRIPTS

The [`examples`](examples) folder contains small Python entry points that
demonstrate the typical workflow end-to-end:

- [`toy_dataset.py`](examples/toy_dataset.py) builds a synthetic
  `GraphDataSet` by generating two water-like metastable states, optionally
  saving the graphs to disk for reuse.
- [`train_deeptda.py`](examples/train_deeptda.py) fits a
  [`GraphDeepTDA`](cvs/supervised/deeptda.py) collective variable with the GVP
  architecture using PyTorch Lightning, logging metrics and storing a
  checkpoint.
- [`analyze_sensitivity.py`](examples/analyze_sensitivity.py) reloads the
  trained model checkpoint and computes atom-wise sensitivities to highlight
  which atoms drive the learned CV.
- [`train_committor.py`](examples/train_committor.py) demonstrates how to train
  a [`GraphCommittor`](cvs/committor/committor.py) model that learns the
  committor function directly from graph data.
- [`compute_committor_bias.py`](examples/compute_committor_bias.py) loads a
  trained committor, evaluates the Kolmogorov bias, and updates dataset weights
  for use in biased simulations or reweighting workflows.

## DEPENDENCIES

- `pytorch_geometric` >= 2.5
- `matscipy`
- `mdtraj`
