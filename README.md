# **MASET-Net: Enhancing Medical Image Segmentation with Multi-level Attention and Transformer-based Convolutional Neural Networks**  

## **Datasets**  
The following datasets are used to evaluate the performance of the MASET framework::  

- **CVC-ClinicDB**  
  - 📂 [CVC-ClinicDB Dataset](https://www.kaggle.com/datasets/sjhuang26/cvcclinicdb)  
  - A collection of 612 high-resolution images from colonoscopy videos, annotated for polyp segmentation tasks.  

- **Kvasir-SEG**  
  - 📂 [Kvasir-SEG Dataset](https://www.kaggle.com/datasets/andrewmvd/kvasir-seg)  
  - Contains 1000 polyp images from endoscopic procedures with corresponding ground truth segmentation masks.  

- **ISIC-2018**  
  - 📂 [ISIC-2018 Challenge Dataset](https://challenge2018.isic-archive.com/)  
  - A large dataset for skin lesion analysis, containing 2594 dermoscopic images with lesion segmentation masks.  

## **Introduction**  
**MASET (Multi-scale Attention-based Segmentation Transformer)** is a deep learning framework for medical image segmentation. It enhances standard U-Net architectures with SE block, Attention gates and transformer-based encodings for superior performance.  

## **Related Works**
Some exmaples of related works included in this paper:
- Ronneberger, O., Fischer, P., Brox, T.: U-Net: Convolutional networks for biomedical image segmentation. Med. Image Comput. Comput. Assist. Interv., 234-241 (2015). https://doi.org/10.1007/978-3-319-24574-4_28
- Oktay, O., Schlemper, J., Folgoc, L. L., et al.: Attention U-Net: Learning where to look for the pancreas (2018). http://dx.doi.org/10.48550/arXiv.1804.03999
- Zhang, L., Chen, H., Wang, Y.: TransAttUNet: Multi-level attention-guided U-Net with Transformer for medical image segmentation. J. Med. Imaging Comput. Vis. (2024), 9(3), 245-258. https://doi.org/10.1109/TETCI.2023.3309626

## **Installation & Setup**  

### **Requirements**  
To install dependencies, run:  

```bash
pip install -r requirements.txt
```
### **Usage Instructions**
- To utilize our code, you need to ensure the following setup within the PyTorch framework: please refer to the requirements.txt for further information.
- To successfully load and preprocess the data, first download the datasets from the provided link to your local system. Next, add the paths to your working Python console, and then execute the dataset.py and preprocess.py files in that order.
- Execute the model.py file to initialize the model.
- To assess the performance of the model, you can execute the evaluation.py file.
  
## **License**
📜 This project is licensed under the MIT License.
