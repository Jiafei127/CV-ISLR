# WWW25@CV-ISLR

This repository contains our implementation for the Cross-View Isolated Sign Language Recognition (CV-ISLR) task submitted to the WWW 2025 competition. Our approach combines **Ensemble Learning** and **Video Swin Transformer (VST)** modules to address the challenges of cross-view sign language recognition. The framework is built on top of the [MMAction2 v1.2.0](https://github.com/open-mmlab/mmaction2) library.

---

## **Main Contributions**
1. **Ensemble Learning Integration**:  
   We integrate Ensemble Learning into the CV-ISLR framework, enhancing robustness and generalization to effectively handle viewpoint variability.

2. **Multi-Dimensional VST Blocks**:  
   We utilize VST blocks of varying sizes (Small, Base, Large) for both RGB and Depth videos, capturing features at multiple levels of granularity to improve recognition accuracy.

---

## **Installation**

To set up the environment, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Jiafei127/CV-ISLR.git
   cd CV-ISLR
   
2. Install dependencies:

   ```bash
   conda create -n cv_islr python=3.8 -y
   conda activate cv_islr
   conda install pytorch torchvision -c pytorch  # This command will automatically install the latest version PyTorch and cudatoolkit, please check whether they match your environment.
   pip install -U openmim
   mim install mmengine
   mim install mmcv
   mim install mmdet  
   mim install mmpose 

3. Install MMAction2 v1.2.0:

   ```bash
   pip install -v -e .

Below is a markdown file for the CV-ISLR competition code repository. It adheres to GitHub's best practices for a well-documented repository.

---

## **Training**

To train the models for RGB and Depth inputs:

1. **Prepare the dataset**: Download and preprocess the [MM-WLAuslan dataset](https://example-dataset-link.com) following the instructions provided in the `dataset/README.md`.

2. **Train the backbone models**:
   ```bash
   python tools/train.py configs/recognition/swin/swin-<file_name>_rgb.py
   python tools/train.py configs/recognition/swin/swin-<file_name>_depth.py
   ```

3. **Save model checkpoints**: After training, checkpoints will be saved in the `work_dirs/` folder.

---

## **Ensemble Learning**

After training the individual models, apply the ensemble strategy:

1. Merge predictions from multiple backbones:
   ```bash
   cd ./ENSEMBLE
   python ensemble.py
   ```
---

## **Performance**

### **Top-1 Accuracy Results**

| Team          | RGB Acc@1 | RGB-D Acc@1 |
|---------------|-----------|-------------|
| VIPL-SLR      | 56.87%    | 57.97%      |
| tonicemerald  | 40.30%    | 33.97%      |
| gkdx2 (Ours)  | 20.29%    | 24.53%      |

_Table 1: The top-3 results for CV-ISLR on RGB and RGB-D tracks._

| Backbone    | RGB-based | Depth-based | RGB-D-based |
|-------------|-----------|-------------|-------------|
| VST-Small   | 14.84%    | 14.01%      | -           |
| VST-Base    | 17.51%    | 16.46%      | -           |
| VST-Large   | 17.04%    | 17.58%      | -           |
| Ensemble    | 20.29%    | -           | 24.53%      |

_Table 2: Experimental results for RGB and RGB-D tracks on different backbones._

---

## **Acknowledgements**

This project is built on the [MMAction2](https://github.com/open-mmlab/mmaction2) framework and utilizes the [MM-WLAuslan dataset](https://example-dataset-link.com). We thank the developers for their contributions to open-source tools and datasets.

---
