# TabM PyTorch Reference

This repository provides a lightweight, fully self-contained PyTorch implementation of several tabular architectures introduced by Gorishniy et al. The library is published as `tabularx` and currently has the following implementaitons:

- Tuned MLP and tabular ResNet baselines from “Revisiting Deep Learning Models for Tabular Data” (arXiv:2106.11959), covering Equations (1) and (2) in the paper.

- TabM and TabM-mini from “TabM: Advancing Tabular Deep Learning with Parameter-efficient Ensembling” (2025).

The code mirrors the original designs: BatchEnsemble adapters for TabM, residual blocks with BatchNorm/Dropout for the tabular ResNet, and the plain multilayer perceptron baseline. Post-hoc pruning utilities allow keeping only the best-performing TabM submodels for inference.

## Quickstart

> Requires Python 3.10+.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pytest
```

The default dependency set includes PyTorch, `typing_extensions`, and `pytest` for the accompanying unit tests.

## Basic usage

```python
import torch
from tabularx import TabMConfig, TabM, MLPConfig, MLP, ResNetConfig, TabularResNet

tabm_cfg = TabMConfig(input_dim=20, output_dim=1, hidden_dim=256, num_layers=4, dropout=0.2, ensemble_size=32)
tabm = TabM(tabm_cfg)
logits = tabm(torch.randn(32, tabm_cfg.input_dim))

mlp = MLP(MLPConfig(input_dim=20, output_dim=1))
resnet = TabularResNet(ResNetConfig(input_dim=20, output_dim=1, hidden_dim=512, num_blocks=4))
```

Pass `return_submodel_outputs=True` to TabM’s `forward` to also obtain the per-member logits. Call `prune_submodels([...])` after validation to keep only a subset of members, mirroring the greedy selection strategy discussed in Section 5.2 of the paper.

See `examples/basic_usage.py` for runnable scripts covering all exported models.

## Notebooks

- `notebooks/tabm_mini_vs_mlp.ipynb`: end-to-end Lightning workflow that compares TabM-mini and a plain MLP with the same hidden size on a toy classification dataset. It uses AdamW plus a cosine LR schedule with a 25% linear warmup to highlight how to integrate the models into full training loops. Install the optional notebook dependencies with `pip install -e .[notebook]`.
