# 🧠 Machine Learning Projects Collection

This repository contains several machine learning projects using different data types: tabular, image, audio, and text. Each project is implemented in a separate Jupyter Notebook.

---

## 📁 Notebooks Overview

### 5. `05_Tabular_Data_Classification.ipynb`
**Rice Type Classification**  
Binary classification of rice grain types using structured tabular data. Dataset contains modified labels: Jasmine = 1, Gonen = 0.

---

### 6. `06_Image_Classification.ipynb`
**Animal Faces Classification**  
Image classification based on the AFHQ dataset with 16,130 high-resolution animal face images from three categories:
- Cat
- Dog
- Wildlife

---

### 7. `07_Pre-trained_Models_Image_Classification.ipynb`
**Bean Leaf Lesion Detection (GoogLeNet)**  
Image classification using a pre-trained GoogLeNet model to detect three leaf states:
- Healthy
- Angular leaf spot
- Bean rust

---

### 8. `08_Audio_Classification.ipynb`
**Quran Recitation Audio Classification**  
Audio classification task using Quran recitations recorded by various reciters. The dataset includes WAV files organized by speaker, recorded in different acoustic conditions.

---

### 9. `09_Text_Classification.ipynb`
**Sarcasm Detection in News Headlines (BERT)**  
Text classification using a fine-tuned BERT model (`google-bert/bert-base-uncased`) to detect sarcasm in news headlines. The dataset includes:
- Sarcastic headlines from *The Onion*
- Real headlines from *HuffPost*

---

### 10. `10_Multi_Task_Learning.ipynb`  
**Age, Gender & Race Estimation (UTKFace, Multi-Task Learning)**  
Image-based multi-task learning pipeline using the UTKFace dataset to simultaneously predict three facial attributes:
- Age (regression)
- Gender (binary classification)
- Race (multi-class classification)

The model is built on a fine-tuned ResNet-50 (`timm/resnet50`) and uses tailored image transformations for improved performance across tasks.

---

## 🛠 Technologies

- Python 3.x  
- Jupyter Notebook  
- PyTorch (depending on the notebook)  
- BERT (Hugging Face Transformers)  
- GoogLeNet (TorchVision)

---


