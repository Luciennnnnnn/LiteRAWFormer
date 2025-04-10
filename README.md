<div align="center">

# Winner solution for NTIRE 2025 RAW Image SR Challenge

Tianyu Zhang<sup>*</sup>, Xin Luo<sup>*</sup>, Yeda Chen, Dong Liu

<sup>*</sup> Equal contribution

[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](#license)

</div>

<br>

## 1. Installation

```bash
# create a virtual environment [Recommended but optional]
conda create -n ntire2025_ustc_vidar python=3.11
source activate ntire2025_ustc_vidar

# Install necessities
# In root/
bash install.sh
```

## 2. Inference
- Download our [pretrained models](https://drive.google.com/drive/folders/1iGS32Qux9mr4SJt2Zb7XTv4Xq6zLEy6I?usp=drive_link), and place the contents in pretrained_models/
- Download the [validation inputs](https://drive.google.com/file/d/1KF3lCrFZua4hGl9_4Km2uOAnWAv1SjjB/view?usp=sharing), and place it in datasets/RAWSR/
- Running inference with following command, the results are saved in results/RAWSR/val_out
    ```bash
    # in root directory
    bash inference.sh
    ```
- To evaluate the results, you should register [NTIRE 2025 RAW Restoration Challenge](https://codalab.lisn.upsaclay.fr/competitions/21644#learn_the_details) and upload your results to the platform.

## 3. Training
- Download the [Training set](https://drive.google.com/file/d/1rUno3LXfGw013g1EfUvPX1bbpBMyLZEU/view?usp=sharing), and place it in datasets/RAWSR/
- Running training with following command.
    ```bash
    # in root directory
    bash train.sh # use all available GPUs
    CUDA_VISIBLE_DEVICES=0,1 bash train.sh # use the first two GPUs
    ```

