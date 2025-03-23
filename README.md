
# 🚀 Machine Learning Based Object Detection in Foggy and Rainy Conditions

## 🌟 Overview
This project implements an advanced object detection system specifically designed for challenging weather conditions such as fog and rain. It leverages **YOLOv8** as the base model, enhanced with custom preprocessing techniques and architectural improvements to ensure robust detection in adverse environments.Visibility is significantly reduced in foggy and rainy conditions, making traditional object detection methods unreliable. The goal is to develop an AI-based detection system that enhances vehicle safety under such conditions.

## 🔥 Key Features
- **🌫️ Weather-Adaptive Object Detection:** Optimized for foggy and rainy conditions.
- **🖼️ Advanced Image Preprocessing:** Uses dehazing and contrast enhancement techniques.
- **📝 Labeled Data: Annotations created using CVAT and MakeSense.
- **📊 Support for Diverse Weather Datasets:** Includes **Foggy Cityscapes and Some Other dataset taken from Kaggle** datasets.
- **⚡ Real-Time Inference:** Optimized for fast and efficient predictions.
- **📱 Edge Deployment Optimizations:** Lightweight model versions for IoT and embedded systems.

## ⚙️ Setup Instructions
###  Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt PyTorch Ultralytics
```

### 2️⃣ Download and Prepare Datasets
Ensure you have the required datasets and place them in the `data/` directory:
- **🌫️ Foggy Cityscapes** (for urban scenes under foggy conditions)
- **☔ RESIDE** (for dehazing research)
- **🌧️ Rainy COCO** (for object detection in rainy environments)

###  3️⃣ Data Preprocessing
Run the preprocessing script to clean and enhance images:
```bash
python src/preprocessing/prepare_data.py
```

###  4️⃣ Model Training
Train the model using the prepared dataset:
```bash
python src/training/train.py --config configs/yolov8_config.yaml
```

###  5️⃣ Inference
Run inference on an input image:
```bash
python src/inference/inference.py --input path/to/image --weights path/to/weights
```

## 🏗️ Model Enhancements
The detection pipeline integrates several enhancements to improve accuracy under low-visibility conditions:
- 📝 Labeled Data: Annotations created using CVAT and MakeSense.
- **🔺 Feature Pyramid Network (FPN)**: Enhances multi-scale feature detection.
- **🧠 Attention Mechanisms**: Improves object localization in low-contrast conditions.
- **📱 Lightweight Model Variants**: Optimized versions for edge computing and real-time applications.

## 📦 Requirements
Python 3.8+ Python NVIDIA GPU (recommended) CUDA Webcam or video source See requirements.txt for full package list

## 📝 Notes
- ⚡ Auto GPU Detection: Uses CUDA if available, falls back to CPU
-  🔧 Configurable Threshold: Adjust confidence in code (default: 0.25)
-  🌐 Pretrained Models: Automatically downloads COCO weights on first run
- 📈 Performance: Processing times ranged from 13.6ms to 102.4ms per image. The model handled various image sizes well
  
## 🤝 Contributors
- **👨‍💻 Project Lead:** Arrol, Harsh
- **👥 Team Members:** Vinit , Pradeep



