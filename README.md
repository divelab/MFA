### Minimal Frame Averaging

---

Official code repository of paper [Equivariance via Minimal Frame Averaging for More Symmetries and Efficiency]().

Currently, we are still organizing codes for different experiments. In this code base, we only provide the equivariance error test for all the group included in the paper, which suffices to show our idea and corresponding algorithm. These tests are

- `euclidean`: Equivariance error test of $O(d)/SO(d)/E(d)/SE(d)$
- `degenerate`: Equivariance error test of $O(d)/SO(d)/E(d)/SE(d)$ for degenerate point clouds
- `algebraic`: Equivariance error test of $O(1,d-1)$ and $SO(1,d-1)$
- `complex`: Equivariance error test of $U(d)$ and $SU(d)$
- `special`: Equivariance error test of $SL(d,\mathbb{R})$ and $GL(d,\mathbb{R})$
- `point_group`: Equivariance error test of $S_n$ and $S_n \times O(d)$

By directly running the corresponding Python files, the error will be printed to the console. Note that the error is listed in the columns of a matrix in an order of “Plain, MFA, FA, SFA” or just “Plain, MFA” if there is only two columns, where “Plain” denotes no frame averaging is used, “MFA” denotes our minimal frame averaging method, “FA” denotes original frame averaging method [1] and “SFA” denotes stochastic frame averaging method [2].

---

[1] Puny, Omri, et al. "Frame averaging for invariant and equivariant network design." *arXiv preprint arXiv:2110.03336* (2021).

[2] Duval, Alexandre Agm, et al. "Faenet: Frame averaging equivariant gnn for materials modeling." *International Conference on Machine Learning*. PMLR, 2023.
