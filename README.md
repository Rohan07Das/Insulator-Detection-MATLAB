# 🔌 Outdoor Insulator Fault Detection and Classification using YOLOv4 and SqueezeNet (MATLAB & Simulink)

This project presents a real-time system for detecting and classifying faults in outdoor transmission line insulators using **YOLOv4** for object detection and **SqueezeNet** for fault classification. The system is implemented and evaluated using **MATLAB R2023a** and integrated with **Simulink** for real-time deployment readiness.

---

## 📖 Project Overview

Power system reliability depends heavily on the health of transmission line insulators. Faults like cracks, flashovers, and contamination can lead to serious electrical failures if not detected in time. This project proposes an intelligent, automated solution for **real-time insulator fault detection and classification** using computer vision and deep learning.

The workflow involves:

- **Detecting insulators and fault regions** using a trained **YOLOv4** model based on the CSPDarknet53 backbone.
- **Classifying the fault types** using a fine-tuned **SqueezeNet** deep learning classifier.
- **Visualizing and evaluating** performance through precision-recall metrics, confusion matrices, and Simulink integration.
- **Testing the system in real-time simulation** via Simulink, with future deployment potential on edge devices like **Raspberry Pi** or **UAVs**.

The system is trained and validated on a labeled dataset of outdoor insulator images derived from the **CPLID** dataset, with annotations converted from Pascal VOC to YOLO and MATLAB format.

---

## 📂 Project Structure

Insulator-Detection/
├── models/ # Trained YOLOv4 and SqueezeNet models
├── src/ # MATLAB scripts and Simulink models
├── data/ # Input images and annotation files
├── results/ # Output figures, plots, and reports
├── utils/ # Helper functions (pre/post-processing)
├── README.md # Project overview


## 🧠 Key Components

- **YOLOv4 (CSPDarknet53 backbone)**  
  Trained to detect insulators and localize defects such as flashover, broken sheds, and contamination.

- **SqueezeNet Classifier**  
  Used to classify the type of defect in the detected insulator region.

- **Post-Processing**  
  Includes bounding box refinement, label overlay, and result visualization.

- **Simulink Integration**  
  System integrated into Simulink for real-time simulation and hardware deployment (e.g., Raspberry Pi).

---

## 📊 Results

- **Precision**, **Recall**, and **mAP** evaluated on custom test set
- **Confusion Matrix** comparing YOLO detections vs. SqueezeNet classification
- Detected and classified output images saved in `results/`

---

## 📁 Dataset

- Based on **CPLID Dataset**  
  Contains labeled images of transmission line insulators with various defects.
- Format:
  - Images: `.jpg`
  - Annotations: Pascal VOC `.xml` (converted to YOLO `.txt` and `.mat` for training)


🔧 Requirements

-MATLAB R2023a,Simulink
-Computer Vision Toolbox and Computer Vision Toolbox for YOLOV4
-Deep Learning Toolbox
-Image Processing Toolbox
-Raspberrypi for MATLAB & Simulink(if you want to deploy)

## 📂 Download Required Files

Due to GitHub's file size restrictions, large files are hosted on Google Drive. Please download them using the links below and place them in the appropriate folders in your project.

| File/Folder                          | Description                              | Download Link |
|--------------------------------------|------------------------------------------|----------------|
| 📁 Dataset Folder (`/data/`)         | Raw image dataset for training/testing   | [Open Folder](https://drive.google.com/drive/folders/1NvPBhwuCmfkadpRA0LlydQ0CM1jdPdLv?usp=drive_link) |
| 🧠 SqueezeNet Classifier (`SqueezeNetClassifier.mat`) | Trained classifier model | [Download](https://drive.google.com/uc?export=download&id=1Fm4JyxSlSvBNbyzhB3_i5itVZjrR4YtL) |
| 🤖 YOLOv4 Model (`YOLOv4.mat`)       | Trained object detection model           | [Download](https://drive.google.com/uc?export=download&id=1RJfhIhkHa_Bv6dNnRDacLDTQIE22if5P) |

> ⚠️ After downloading:
> - Place the `.mat` model files inside a `models/` directory.
> - Copy all dataset images and annotations from the Drive folder into a `data/` directory in your project.

