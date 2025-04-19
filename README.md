<div align="center">
<h1> RoSPER-Net: Robust Medical Image Segmentation with Spatial Prompting and Cross-Scale Edge Refinement </h1>
</div>

## 🎈 News

- [2025.4.19] Training and inference code released

## ⭐ Abstract

Medical image segmentation is crucial for clinical decision making and treatment planning.
However, it faces two challenges:
First, the structures of salient objects and background details vary significantly in medical images of different modalities.
Second, the misleading co-occurrence of salient and non-salient objects and the noise interference at the edges affect the segmentation accuracy of the model.
To overcome these challenges, we propose RoSPER-Net, a framework designed to enhance medical image segmentation.
RoSPER-Net integrates a Spatial Prompt Encoder (SPE), which generates two complementary prompts using an advanced prompt mechanism to guide the model to focus on the local-global structure of salient objects and understand the overall background information in the image, thereby improving the model's adaptability and segmentation accuracy under different modalities and complex backgrounds.
Plus, our Cross-Scale Edge Enhancement Decoder (CSED) uses noise suppression and edge enhancement mechanisms to suppress non-salient regions and highlight salient regions, thereby improving the model's ability to detect salient objects in complex backgrounds. Comprehensive evaluations of RoSPER-Net on 5 medical image datasets verify its superior performance and versatility, demonstrating its potential in the field of medical image segmentation.

## 🚀 The challenges

<div align="center">
    <img width="400" alt="image" src="asserts/cha1.png?raw=true">
</div>

<div align="center">
    <img width="400" alt="image" src="asserts/cha2.png?raw=true">
</div>

## 📻 Overview

<div align="center">
<img width="800" alt="image" src="asserts/network.png?raw=true">
</div>

Illustration of the overall architecture of RoSPER-Net. (II) SPE is Spatial Prompt Encoder, (III) CSED is Cross-scale Edge Enhancement Decoder. (II.a) PMM is Parallel Mamba Module, (II.b) SPM is Self-prompting Module, (II.c) CPB is Channel-prior Block, (II.d) SPB is Space-prior Block, (III.a) MSFE is Multi-scale Feature Extraction, (III.b) DCT is Dynamic Contour Tracking Module, (III.c) NSB is Noise Suppression Block, (III.d) DADCB is Direction-aware Deformable Convolution Block.


## 📆 TODO

- [x] Release code

## 🎮 Getting Started

### 1. Install Environment

```
conda create -n RoSPERNet python=3.8
conda activate RoSPERNet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
pip install pywt
```

### 2. Prepare Datasets

- Download datasets: ISIC2017 from this [link](https://challenge.isic-archive.com/data/#2017), ISIC2018 from this [link](https://challenge.isic-archive.com/data/#2018), and PH2 from this [link](https://www.dropbox.com/scl/fi/epzcoqeyr1v9qlv/PH2Dataset.rar?rlkey=6mt2jlvwfkditkyg12xdei6ux&e=1), Kvasir from this[link](https://link.zhihu.com/?target=https%3A//datasets.simula.no/downloads/kvasir-seg.zip), and BUSI from this [link](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset).
- Folder organization: put ISIC2017 datasets into ./data/ISIC2017 folder, ISIC2018 datasets into ./data/ISIC2018 folder, and PH2 datasets into ./data/PH2 folder, Kvasir datasets into ./data/Kvasir folder, and BUSI datasets into ./data/BUSI folder.

### 3. Train the RoSPER-Net

```
python train.py --datasets ISIC2018
training records is saved to ./log folder
pre-training file is saved to ./checkpoints/ISIC2018/best.pth
concrete information see train.py, please
```

### 3. Test the RoSPER-Net

```
python test.py --datasets ISIC2018
testing records is saved to ./log folder
testing results are saved to ./Test/ISIC2018/images folder
concrete information see test.py, please
```


## 🖼️ Visualization

<div align="center">
<img width="800" alt="image" src="asserts/compara.png?raw=true">
</div>

Visualization results of ten state-of-the-art methods and RoSPER-Net for different lesions. The red circles indicate areas of incorrect predictions.

## ✨ Quantitative comparison

<div align="center">
<img width="800" alt="image" src="asserts/table.png?raw=true">
</div>

<div align="center">
    Performance comparison with ten SOTA methods on ISIC2017, ISIC2018, PH2, Kvasir and BUSI datasets.
</div>

## 🎫 License

The content of this project itself is licensed under [LICENSE](https://github.com/ILoveESWA/RoSPERNet/blob/main/LICENSE).

