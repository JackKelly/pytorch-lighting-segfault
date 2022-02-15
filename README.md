Minimal code example to demonstrate segfault in pytorch-lighting.

Install the conda environment with `conda env create -f environment.yml` then run `conda activate segfault` 

Run `python pytorch-lightning.py` to show the segfault.

`pytorch.py` is a minimal code example which _does not_ use pytorch lightning, and does _not_ show the segfault.

Reported here:
https://github.com/PyTorchLightning/pytorch-lightning/issues/11925
