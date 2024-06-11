# Minimal Frame Averaging

---

## Overview

Official code repository of paper [Equivariance via Minimal Frame Averaging for More Symmetries and Efficiency](). In this repository, we have provided decorators to convert any non-equivariant/invariant neural network functions into group equivariant/invariant ones. This enables neural networks to handle transformations from various groups such as $O(d), SO(d), O(1,d-1)$, etc.

Currently, we are still organizing codes for different experiments in our paper. In this code base, we currently provide the equivariance error test for all the groups included in the paper, which suffices to show our idea and corresponding algorithms. Stay tuned for more details!




## Installation

**Note**: We recommend that you install the `torch` package separately.

To install the package, use the following command:

```sh
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118
```



## Usage

### Example: $O(d)$-Equivariant/Invariant Decorator

The `od_equivariant_decorator` can be used to wrap any forward function to make it $O(d)$​-equivariant. An example model can be

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define your layers here

    def forward(self, x):
        # Your forward pass implementation
        return x
```

To apply this decorator to this neural network class’s forward function:

```python
import torch.nn as nn
from minimal_frame.group_decorator import od_equivariant_decorator

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define your layers here
        
    @od_equivariant_decorator
    def forward(self, x):
        # Your forward pass implementation
        return x
```

To apply this decorator to a neural network instance's forward function:

```python
# Instantiate your model
model = MyModel()

# Apply the O(d)-equivariant decorator
model.forward = od_equivariant_decorator(model.forward)
```

Similarly, the `od_invariant_decorator` can be used to make the network $O(d)$-invariant.



## Supported Groups

This repository provides a collection of decorators to enforce equivariance or invariance properties in neural network models. Equivariance ensures that the output of the model transforms in the same way as the input under a given group of transformations. Invariance ensures that the output remains unchanged under such transformations.

These decorators are model-agnostic and can be applied to any model with input and output shapes of $n\times d$ for equivariance or with input shape of $n\times d$ and output shape of $1$ for invariance. A comprehensive equivariance analysis over various groups can be found [here](https://github.com/divelab/MFA/blob/main/tests/equivariance_test.ipynb), including


- Orthogonal Group $O(d)$ 
- Special Orthogonal Group $SO(d)$
- Euclidean Group $E(d)$
- Special Euclidean Group $SE(d)$
- Unitary Group $U(d)$
- Special Unitary Group $SU(d)$
- Lorentz Group $O(1,d-1)$
- Proper Lorentz Group $SO(1,d-1)$
- General Linear Group $GL(d,\mathbb{R})$
- Special Linear Group $SL(d,\mathbb{R})$
- Affine Group $Aff(d,\mathbb{R})$
- Permutation Group $S_n$
- Direct product between Permutation Group $S_n$ and other groups
  - $S_n \times O(d)$
  - $S_n \times SO(d)$
  - $S_n \times O(1,d-1)$​

Additionally, the $S_n$-equivariant/invariant decorators for undirected graphs are provided for any model with undirected adjacency matrices as input. The corresponding equivariance analysis can be found [here](https://github.com/divelab/MFA/blob/main/tests/graph_equivariance_test.ipynb).





## References

[1] Puny, Omri, et al. "Frame averaging for invariant and equivariant network design." *arXiv preprint arXiv:2110.03336* (2021).

[2] Duval, Alexandre Agm, et al. "Faenet: Frame averaging equivariant gnn for materials modeling." *International Conference on Machine Learning*. PMLR, 2023.

[3] McKay, Brendan D. "Practical graph isomorphism." (1981): 45-87.



## Licence

This project is licensed under the MIT License. Please note that the `nauty` software is covered by its own licensing terms. All rights to the files mentioned in `nauty.py` are reserved by Brendan McKay under the Apache License 2.0.


## Acknowledgements

We gratefully acknowledge the insightful discussions with Derek Lim and Hannah Lawrence. This work was supported in part by National Science Foundation grant IIS-2006861 and National Institutes of Health grant U01AG070112.

## Citation

If you find this work useful, please consider citing:

```bib
@inproceedings{
lin2024equivariance,
title={Equivariance via Minimal Frame Averaging for More Symmetries and Efficiency},
author={Yuchao Lin and Jacob Helwig and Shurui Gui and Shuiwang Ji},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=guFsTBXsov}
}
```