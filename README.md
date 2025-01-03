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
   git clone https://github.com/yourusername/CV_ISLR_WWW2025.git
   cd CV_ISLR_WWW2025
   
2. Install dependencies:
   ```bash
   conda create -n cv_islr python=3.8 -y
   conda activate cv_islr
   pip install -r requirements.txt

3. Install MMAction2 v1.2.0:
   ```bash
   git clone https://github.com/open-mmlab/mmaction2.git
   cd mmaction2
   pip install -e .
