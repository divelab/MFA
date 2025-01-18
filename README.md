# Open Catalyst Project Experiments

## Introduction

This repository contains experiments for the Open Catalyst Project, utilizing the [FAENet framework](faenet/README.md) for model training and evaluation.

## Environment Setup

Please refer to the [FAENet Installation Guide](faenet/README.md#installation) for detailed instructions on setting up the required environment.

## Data Preparation

1. **Download Dataset**: Obtain the dataset (Train, Validation, Test splits totaling 97 GB) from the [OC20 IS2RE Datasets](https://fair-chem.github.io/core/datasets/oc20.html).

2. **Extract Data**: Unzip the downloaded file.

3. **Configure Paths**: Update all data paths in `faenet/configs/tasks/is2re.yaml` to point to the extracted data directories.

## Training

To train the model on the IS2RE-All split:

1. **Configure Model**: Modify the model settings in `faenet/configs/models/faenet.yaml` under the `all` section as needed.

2. **Run Training**:

   ```bash
   SLURM_JOB_ID=0 SCRATCH=./ python main.py --config=faenet-is2re-all --optim.batch_size=256 --frame_averaging=2D --graph_rewiring=remove-tag-0
   ```
3. **Logs**: Training logs are saved in `$SCRATCH/ocp/runs/$SLURM_JOB_ID`. For the above command, logs are stored in `./ocp/runs/0/`.

## Inference

To perform inference using the pretrained model:

1. **Pretrained Configurations**: The pretrained config located at `faenet/ocp/runs/7/config-7.yaml`.

2. **Checkpoints and Logs**: The checkpoint and logs are located in `faenet/ocp/runs/7/checkpoints` and `faenet/ocp/runs/7/logs` respectively.

3. **Run Inference**:

```bash
SLURM_JOB_ID=0 SCRATCH=./ python main.py --config=faenet-is2re-all --optim.batch_size=256 --optim.max_epochs=0 --frame_averaging=2D --graph_rewiring=remove-tag-0 --checkpoint=ocp/runs/7/checkpoints/best_checkpoint.pt --test_ri
```

4. **Results**: The output metrics will resemble the following table:


| Metric / Split          | val_id  | val_ood_cat | val_ood_ads | val_ood_both |
| ----------------------- | ------- | ----------- | ----------- | ------------ |
| energy_mae              | 0.54368 | 0.54150     | 0.62032     | 0.57078      |
| energy_mse              | 0.79252 | 0.78328     | 0.93364     | 0.74007      |
| energy_within_threshold | 0.04326 | 0.04539     | 0.02961     | 0.02970      |
| energy_loss             | 0.23849 | 0.23748     | 0.27202     | 0.25028      |
| total_loss              | 0.23849 | 0.23748     | 0.27202     | 0.25028      |

## Acknowledgments

This repository is based on [RolnickLab/ocp](https://github.com/RolnickLab/ocp). The primary modification is in `faenet/ocpmodels/preprocessing/frame_averaging.py`. All other rights are reserved to the original authors.