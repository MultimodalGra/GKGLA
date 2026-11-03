# GKGLA

Research implementation accompanying **Grounding General Knowledge via Low-rank Residuals and Attention Fusion for Class-Incremental Learning**.

GKGLA adapts a frozen CLIP backbone with Identity-Anchored Low-Rank Residual Projection (ILRP) and Asymmetric Gated Iterative Fusion (AGIF). The Hierarchical Historical Retrieval Mechanism (HHRM) retrieves preceding image-instance states inside AGIF.

## Implementation

| Paper component | Implementation |
| --- | --- |
| GKGLA training | `models/gkgla.py`: `GKGLALearner` |
| GKGLA network | `models/network.py`: `GKGLANet` |
| ILRP | `models/ilrp.py`: `ILRP`, `ILRPResidualBranch` |
| AGIF | `models/agif.py`: `AGIF` |
| HHRM | `models/agif.py`: `HHRM.retrieve` |

ILRP and AGIF are implementation mixins. Their parameters are registered on `GKGLANet`; the training routine assigns learning rates using these parameter names.

## Setup

Use Python 3.10 or newer. Create a virtual environment and install a matching PyTorch/torchvision pair for your hardware using the [PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```sh
python -m venv .venv
```

Windows PowerShell: `.\.venv\Scripts\Activate.ps1`. Linux: `source .venv/bin/activate`.

```sh
python -m pip install -r requirements.txt
python main.py --config configs/cifar.json --check-config
```

The dependency ranges above are not a lockfile for the manuscript experiments.

## Data and weights

`data_root` defaults to `./data`. CIFAR-100 downloads automatically. Other datasets require prepared class-folder splits:

```text
data/
  imagenet-r/{train,test}/<class_name>/<image>
  objectnet/{train,test}/<class_name>/<image>
  cub/{train,test}/<class_name>/<image>
  food101/{train,test}/<class_name>/<image>
  aircraft/{train,test}/<class_name>/<image>
  ucf/{train,test}/<class_name>/<image>
  cars/{train,test}/<class_name>/<image>
  sun/{train,test}/<class_name>/<image>
```

Train/test folders must use identical class names. The ImageFolder alphabetical class order must match `utils/labels.json`. These loaders expect the prepared subsets and splits used by the experiments; arbitrary raw datasets are not equivalent. See the [PROOF data preparation references](https://github.com/LAMDA-CL/PROOF#-requirements). Datasets and weights are not distributed here.

The default backbone is OpenCLIP `ViT-B-16`, pretrained tag `laion400m_e32`. [OpenCLIP](https://github.com/mlfoundations/open_clip) downloads the weights on first use. A local checkpoint can be specified with `--clip-checkpoint /path/to/vit_b_16-laion400m_e32-55e67d44.pt`.

## Training

```sh
python main.py --config configs/cifar.json --data-root ./data --device 0
python main.py --config configs/cifarb50.json --data-root ./data --device 0
```

Two example JSON configurations are provided for CIFAR-100. The `cifar.json` configuration uses ten classes in the first task and ten per subsequent task (the B0 Inc10 protocol); `cifarb50.json` starts with 50 classes. These are usage examples, not verified configurations for reproducing the manuscript tables. Loaders for the other datasets remain available, but their experiment configurations are not included in this release. Logs are written under `logs/gkgla/`. One device per experiment is supported. CPU checks can use `--device cpu --num-workers 0`; full training on CPU is expensive.

`seed` controls class ordering, while `training_seed` controls training randomness and defaults to 1. The optimizer uses parameter-group learning-rate multipliers: residual branches 5, attention 0.5, normalization and gates 0.1, prompts 0.2, historical queries 1, and other trainable parameters 0.3. With `fixed_memory=false`, the total exemplar budget is fixed and the per-class budget is `memory_size // number_of_seen_classes`. These example settings do not implement a fixed 20 exemplars per class throughout training.

`save_stage_checkpoints=true` optionally saves stage snapshots. These are inspection snapshots, not a complete training-resume interface. Exemplar memory and visual prototypes are not included in a resumable format.

## Validation

```sh
python -m compileall -q .
python -m unittest discover -s tests -v
```

Tests check identity preservation, old-branch freezing, gradient flow, and HHRM retrieval using a small synthetic backbone. They do not download weights or datasets. Full benchmark accuracy has not been validated for this release.

## Acknowledgements and release status

This implementation is derived from [PROOF](https://github.com/LAMDA-CL/PROOF). Its upstream acknowledges [PyCIL](https://github.com/G-U-N/PyCIL) and CIL_Survey. GKGLA adds the ILRP and AGIF/HHRM mechanisms described above. Please cite the relevant prior work when using the inherited framework.

**Public redistribution licensing is pending verification.** The redistribution license for the inherited PROOF code has not been established. This repository does not grant a new license to inherited code. Public redistribution requires confirmation of upstream permission and applicable terms.
