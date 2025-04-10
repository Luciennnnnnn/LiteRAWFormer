<div align="center">

# 🏆 Winner Solution for NTIRE 2025 RAW Image Super-Resolution Challenge

Tianyu Zhang<sup>1♠️</sup>, Xin Luo<sup>1♠️</sup>, Yeda Chen<sup>2♠️</sup>, Dong Liu<sup>1♠️</sup>
\textsuperscript{1} University of Science and Technology of China, Hefei, China \\
\textsuperscript{2} Shanghai Shuangshen Information Technology Co., Ltd.
<sup>♠️</sup> Equal contribution

[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](#license)

</div>

<br>

## 📌 Introduction

This repository contains the official implementation of our award-winning solution for the NTIRE 2025 RAW Image Super-Resolution Challenge. Our method achieves state-of-the-art performance with enhanced degradation modeling and efficient architecture design.

## 📦 Installation

```bash
# create a virtual environment [Recommended but optional]
conda create -n lite_rawformer python=3.11
source activate lite_rawformer

# Install all necessary dependencies
# In root/
bash install.sh
```

## :rocket: Inference
- Download our [pretrained models](https://drive.google.com/drive/folders/1yBwFUOOS74O5Okyn58G9te8hfOu--Unl?usp=sharing), and place the contents in pretrained_models/
- Download the [validation inputs](https://drive.google.com/file/d/1KF3lCrFZua4hGl9_4Km2uOAnWAv1SjjB/view?usp=sharing), and place it in datasets/RAWSR/
- Running inference with following command, the results are saved in results/RAWSR/val_out
    ```bash
    # in root directory
    bash inference.sh
    ```
- To evaluate the results, you should register [NTIRE 2025 RAW Restoration Challenge](https://codalab.lisn.upsaclay.fr/competitions/21644#learn_the_details) and upload your results to the platform.

## :boat: Training
- Download the [Training set](https://drive.google.com/file/d/1rUno3LXfGw013g1EfUvPX1bbpBMyLZEU/view?usp=sharing), and place it in datasets/RAWSR/
- Running training with following command.
    ```bash
    # in root directory
    bash train.sh # use all available GPUs
    CUDA_VISIBLE_DEVICES=0,1 bash train.sh # use the first two GPUs
    ```
## :email: Contact
If you have any questions, please open an issue (*the recommended way*) or contact us via 
- xinluo@mail.ustc.edu.cn
- zhangtianyu@mail.ustc.edu.cn

## License
This work is licensed under MIT license. See the [LICENSE](https://github.com/Luciennnnnnn/LiteRAWFormer/blob/main/LICENSE) for details.

## Acknowledgement
Our repository builds upon the excellent framework provided by [accelerate](https://github.com/huggingface/accelerate), and our architecture are inspired by [RBSFormer](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Jiang_RBSFormer_Enhanced_Transformer_Network_for_Raw_Image_Super-Resolution_CVPRW_2024_paper.pdf).