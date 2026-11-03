# GKGLA

Code for **Grounding General Knowledge via Low-rank Residuals and Attention Fusion for Class-Incremental Learning**.

## Requirements

Python 3.10+, PyTorch, and torchvision. Install PyTorch and torchvision for your environment, then run:

```sh
pip install -r requirements.txt
```

## Usage

Two example configurations are provided for CIFAR-100:

```sh
python main.py --config configs/cifar.json --data-root ./data --device 0
python main.py --config configs/cifarb50.json --data-root ./data --device 0
```

CIFAR-100 is downloaded automatically. The default backbone is OpenCLIP ViT-B/16 with LAION-400M pretrained weights.

## Acknowledgements

This code is based on [PROOF](https://github.com/LAMDA-CL/PROOF), which also acknowledges PyCIL and CIL_Survey.

Licensing of the inherited code is pending confirmation.
