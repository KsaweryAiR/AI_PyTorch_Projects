# 🧠 Machine Learning Projects Collection

This repository contains several machine learning projects using different data types: tabular, image, audio, and text. Each project is implemented in a separate Jupyter Notebook.

<img src="logo/Pytorch_logo.png" style="width: 20%; height: 20%;" alt="mapa 1" />   

---

## 📁 Notebooks Overview

### 1. `01_Lightning_CNN.ipynb`  
**CNN Classifier on CIFAR-10 (PyTorch Lightning)**  
Introduction to deep convolutional neural networks using the CIFAR-10 dataset. The notebook covers:
- Basic CNN architecture design in PyTorch
- Training pipeline using PyTorch Lightning
- Evaluation on image classification task with 10 object categories

---

### 2. `02_Lightning_Data_Augmentation.ipynb`  
**Image Classification with Limited Data (Data Augmentation)**  
Image classification using convolutional neural networks in scenarios with limited data availability. The notebook demonstrates how to apply data augmentation techniques to improve model generalization and performance.

---

### 3. `03_Lightning_Classification.ipynb`  
**Extended Image Classification (Cats vs Dogs Dataset)**  
Further exploration of image classification using convolutional neural networks, applied to the Cats vs Dogs dataset. The notebook expands on previous concepts with enhanced data handling and model training techniques.

---

### 4. `04_Lightning_Segmentation.ipynb`  
**Image Segmentation (Dog Mask Prediction)**  
Neural networks applied to image segmentation tasks. The goal is to input an image and output a segmentation mask highlighting the dog in the picture.

---

### 5. `05_Neural_network_optimization_methods.ipynb`  
**Neural Network Optimization Techniques (with ONNX)**  
Overview of common methods used to accelerate computations and optimize neural network models, including:
- Quantization and pruning
- Layer fusion and weight clustering
- Knowledge distillation and low-rank factorization
- Parallel and asynchronous processing  
The notebook also demonstrates exporting and optimizing models using **ONNX**.

---

### 6. `06_Tabular_Data_Classification.ipynb`
**Rice Type Classification**  
Binary classification of rice grain types using structured tabular data. Dataset contains modified labels: Jasmine = 1, Gonen = 0.

---

### 7. `07_Image_Classification.ipynb`
**Animal Faces Classification**  
Image classification based on the AFHQ dataset with 16,130 high-resolution animal face images from three categories:
- Cat
- Dog
- Wildlife

---

### 8. `08_Pre-trained_Models_Image_Classification.ipynb`
**Bean Leaf Lesion Detection (GoogLeNet)**  
Image classification using a pre-trained GoogLeNet model to detect three leaf states:
- Healthy
- Angular leaf spot
- Bean rust

---

### 9. `09_Audio_Classification.ipynb`
**Quran Recitation Audio Classification**  
Audio classification task using Quran recitations recorded by various reciters. The dataset includes WAV files organized by speaker, recorded in different acoustic conditions.

---

### 10. `10_Text_Classification.ipynb`
**Sarcasm Detection in News Headlines (BERT)**  
Text classification using a fine-tuned BERT model (`google-bert/bert-base-uncased`) to detect sarcasm in news headlines. The dataset includes:
- Sarcastic headlines from *The Onion*
- Real headlines from *HuffPost*

---

### 11. `11_Multi_Task_Learning.ipynb`  
**Age, Gender & Race Estimation (UTKFace, Multi-Task Learning)**  
Image-based multi-task learning pipeline using the UTKFace dataset to simultaneously predict three facial attributes:
- Age (regression)
- Gender (binary classification)
- Race (multi-class classification)

The model is built on a fine-tuned ResNet-50 (`timm/resnet50`) and uses tailored image transformations for improved performance across tasks.

---
### 12. `12_YOLO_Object_Detection_Bounding_Boxes.ipynb`  
**Vehicle Detection with YOLOv5 (Bounding Boxes)**  
Object detection using a YOLOv11s model (`yolo11s.pt`) to locate and classify vehicles in images. The notebook demonstrates:
- Loading and running a pre-trained YOLO model
- Drawing bounding boxes around detected vehicles
- Interpreting detection confidence and class predictions

---
