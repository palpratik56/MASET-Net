
# **MASET-Net: A Modular Attention-Enhanced Encoder–Decoder Framework for Medical Image Segmentation**

This repository provides the official PyTorch implementation of **MASET-Net**, a modular attention-enhanced encoder–decoder architecture for medical image segmentation. The framework is designed to improve segmentation accuracy through structured multi-scale feature interaction and selective attention integration, while maintaining computational efficiency and architectural interpretability.

MASET-Net has been evaluated on multiple public benchmark datasets and demonstrates robust performance across diverse medical imaging modalities.

---

## **Datasets**

The proposed framework is evaluated using the following publicly available datasets:

### **CVC-ClinicDB**

* 📂 Dataset link: [https://www.kaggle.com/datasets/sjhuang26/cvcclinicdb](https://www.kaggle.com/datasets/sjhuang26/cvcclinicdb)
* Description: A benchmark dataset for colon polyp segmentation consisting of 612 high-resolution colonoscopy images with pixel-level annotations.

### **Kvasir-SEG**

* 📂 Dataset link: [https://www.kaggle.com/datasets/andrewmvd/kvasir-seg](https://www.kaggle.com/datasets/andrewmvd/kvasir-seg)
* Description: A dataset of 1,000 endoscopic images containing polyp regions with corresponding ground truth segmentation masks.

### **ISIC-2018**

* 📂 Dataset link: [https://challenge2018.isic-archive.com/](https://challenge2018.isic-archive.com/)
* Description: A large-scale dermoscopic image dataset for skin lesion segmentation, containing 2,594 annotated images.

---

## **Overview of MASET-Net**

**MASET-Net (Multi-scale Attention-based Segmentation Network)** is a modular encoder–decoder framework that integrates convolutional feature extraction with lightweight attention mechanisms.
The architecture introduces three core components:

* **Adaptive Scale Interaction Block (ASIB):** Enhances multi-scale feature aggregation within the encoder.
* **Transformer-Aware Attention Core (TAAC):** Models long-range dependencies at the bottleneck using efficient attention mechanisms.
* **Feature-Adaptive Attention Bridge (FAAB):** Refines skip-connection features to suppress redundant activations and emphasize semantically relevant regions.

This targeted integration of attention improves boundary delineation and foreground–background discrimination without relying on heavy transformer encoders.

---

## **Related Work**

The design of MASET-Net is inspired by and compared against several foundational and recent segmentation frameworks, including:

* Ronneberger, O., Fischer, P., Brox, T.: *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI, 2015.
* Oktay, O., Schlemper, J., Folgoc, L. L., et al.: *Attention U-Net: Learning Where to Look for the Pancreas*. arXiv:1804.03999.
* Zhang, L., Chen, H., Wang, Y.: *TransAttUNet: Multi-level Attention-guided U-Net with Transformer for Medical Image Segmentation*. IEEE TETCI, 2024.

---

## **Installation and Setup**

### **Requirements**

All dependencies are listed in the `requirements.txt` file.
To install them, run:

```bash
pip install -r requirements.txt
```

---

## **Usage Instructions**

1. **Dataset Preparation**

   * Download the datasets from the links provided above.
   * Update the dataset paths in the configuration or data loader scripts accordingly.
   * Run `dataset.py` followed by `preprocess.py` to prepare the data.

2. **Model Initialization**

   * Execute `model.py` to initialize the MASET-Net architecture.

3. **Training**

   * Run `train.py` to train the model.
   * Training hyperparameters such as the number of epochs, optimizer settings, and learning rate can be adjusted within the script.

4. **Evaluation**

   * Execute `evaluation.py` to compute segmentation metrics on the validation or test set.

---

## **References**

Selected references relevant to the proposed framework include:

* Yao, X., Zhu, Z., Kang, C., et al.: *AdaD-FNN for Chest CT-based COVID-19 Diagnosis*. IEEE TETCI, 2022.
* Zhu, X., Yao, X., Zhang, J., et al.: *TMSDNet: Transformer with Multi-scale Dense Network for 3D Reconstruction*. Computer Animation and Virtual Worlds, 2024.
* Bernal, J., Sánchez, F. J., Fernández-Esparrach, M. G., et al.: *WM-DOVA Maps for Accurate Polyp Highlighting in Colonoscopy*. Computerized Medical Imaging and Graphics, 2015.
* Zheng, S., Lu, J., Zhao, H., et al.: *Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers*. CVPR, 2021.

---

## **Citation**

If you find this work useful, please consider citing the corresponding paper:

> *[Citation details to be added upon publication]*
