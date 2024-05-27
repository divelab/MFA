# Minimal Frame Averaging

---

## Overview

Official code repository of paper [Equivariance via Minimal Frame Averaging for More Symmetries and Efficiency](). In this repository, we have provided decorators to convert any non-equivariant/invariant neural network functions into group equivariant/invariant ones. This enables neural networks to handle transformations from various groups such as $O(d), SO(d), O(1,d-1)$, etc.



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
- Symmetric Group $S_n$
- Direct product between Symmetric Group $S_n$ and other groups
  - $S_n \times O(d)$
  - $S_n \times SO(d)$
  - $S_n \times O(1,d-1)$



Currently, we are still organizing codes for different experiments. In this code base, we only provide the equivariance error test for all the groups included in the paper, which suffices to show our idea and corresponding algorithms. Stay tuned for more details!

---

[1] Puny, Omri, et al. "Frame averaging for invariant and equivariant network design." *arXiv preprint arXiv:2110.03336* (2021).

[2] Duval, Alexandre Agm, et al. "Faenet: Frame averaging equivariant gnn for materials modeling." *International Conference on Machine Learning*. PMLR, 2023.
