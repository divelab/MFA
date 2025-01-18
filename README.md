# $n$-Body Experiments

## Introduction

This repository contains experiments for the $n$-Body problem, leveraging the [EGNN framework](https://github.com/vgsatorras/egnn) for model training and evaluation.

## Getting Started

### Navigate to the `nbody` Directory

```bash
cd nbody
```

### Environment Setup

Extend the environment from the `main` branch by adding `torch-geometric` and its dependencies.

1. **Install PyTorch Geometric Dependencies**:

   ```bash
   pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.1%2Bcu118.html
   ```

2. **Install PyTorch Geometric**:

   ```bash
   pip install torch-geometric
   ```

3. **Install Remaining Dependencies**:

   ```bash
   pip install tqdm matplotlib
   ```

## Training

To train the model, execute the following command:

```bash
python nbody.py --exp_name <experiment_name>
```

- **Parameters**:

  - `--exp_name`: Specify a name for the experiment (e.g., `exp`).

- **Outputs**:

  - Training logs and the best checkpoint will be saved in `n_body_system/logs/<experiment_name>/`.

  **Example**:

  ```bash
  python nbody.py --exp_name exp
  ```

  This will save logs and checkpoints to `n_body_system/logs/exp/`.

## Inference

To perform inference using a trained model:

1. **Specify the Checkpoint Path**: Provide the path to the trained model checkpoint (e.g., `n_body_system/logs/exp/model.pth`).

2. **Run Inference**:

   ```bash
   python nbody.py --exp_name <experiment_name> --checkpoint <checkpoint_path> --inference
   ```

   **Example** (Use our provided checkpoint):

   ```bash
   python nbody.py --exp_name exp --checkpoint n_body_system/logs/exp/model.pth --inference
   ```

## Acknowledgments

This repository is based on the [EGNN repository](https://github.com/vgsatorras/egnn). All original rights are reserved to the authors of the EGNN project.
