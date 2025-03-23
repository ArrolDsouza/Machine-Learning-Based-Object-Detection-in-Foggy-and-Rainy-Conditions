
# 🚀 Machine Learning Based Object Detection in Foggy and Rainy Conditions

## 🌟 Overview
This project implements an advanced object detection system specifically designed for challenging weather conditions such as fog and rain. It leverages **YOLOv8** as the base model, enhanced with custom preprocessing techniques and architectural improvements to ensure robust detection in adverse environments.Visibility is significantly reduced in foggy and rainy conditions, making traditional object detection methods unreliable. The goal is to develop an AI-based detection system that enhances vehicle safety under such conditions.

## 🔥 Key Features
- **🌫️ Weather-Adaptive Object Detection:** Optimized for foggy and rainy conditions.
- **🖼️ Advanced Image Preprocessing:** Uses dehazing and contrast enhancement techniques.
- **📊 Support for Diverse Weather Datasets:** Includes **Foggy Cityscapes, RESIDE, and Rainy COCO** datasets.
- **⚡ Real-Time Inference:** Optimized for fast and efficient predictions.
- **📱 Edge Deployment Optimizations:** Lightweight model versions for IoT and embedded systems.

## 📂 Project Structure
```
├── 📁 data/                    # Dataset storage
│   ├── 🌫️ foggy_cityscapes/
│   ├── ☔ reside/
│   └── 🌧️ rainy_coco/
├── 📁 src/
│   ├── 🛠️ preprocessing/       # Image preprocessing modules
│   ├── 🧠 models/              # Model architectures and enhancements
│   ├── 🎯 training/            # Training scripts and utilities
│   ├── 🔍 inference/           # Inference and evaluation scripts
│   └── ⚙️ utils/               # Helper functions
├── ⚙️ configs/                 # Configuration files
├── 📓 notebooks/               # Jupyter notebooks for analysis and visualization
├── 📊 results/                 # Training results, logs, and model checkpoints
└── 📜 requirements.txt         # Dependencies list
```

## ⚙️ Setup Instructions
### 1️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download and Prepare Datasets
Ensure you have the required datasets and place them in the `data/` directory:
- **🌫️ Foggy Cityscapes** (for urban scenes under foggy conditions)
- **☔ RESIDE** (for dehazing research)
- **🌧️ Rainy COCO** (for object detection in rainy environments)

### 4️⃣ Data Preprocessing
Run the preprocessing script to clean and enhance images:
```bash
python src/preprocessing/prepare_data.py
```

### 5️⃣ Model Training
Train the model using the prepared dataset:
```bash
python src/training/train.py --config configs/yolov8_config.yaml
```

### 6️⃣ Inference
Run inference on an input image:
```bash
python src/inference/inference.py --input path/to/image --weights path/to/weights
```

## 🏗️ Model Enhancements
The detection pipeline integrates several enhancements to improve accuracy under low-visibility conditions:
- **🌁 DehazeNet**: Preprocessing network for removing fog and haze.
- **🔺 Feature Pyramid Network (FPN)**: Enhances multi-scale feature detection.
- **🧠 Attention Mechanisms**: Improves object localization in low-contrast conditions.
- **📱 Lightweight Model Variants**: Optimized versions for edge computing and real-time applications.

## 📈 Performance Metrics
The model is evaluated using the following metrics:
- **🎯 mAP (Mean Average Precision)**: Measures detection accuracy.
- **📏 IoU (Intersection over Union)**: Evaluates localization precision.
- **❌ False Positive Rate (FPR)**: Ensures minimal incorrect detections.
- **⚡ Inference Speed (FPS)**: Optimized for real-time applications.

## 🔮 Future Enhancements
- **🛠️ Domain Adaptation:** Improve generalization to unseen weather conditions.
- **🎨 Synthetic Data Augmentation:** Use GANs to generate diverse training samples.
- **🤖 Self-Supervised Learning:** Reduce dependency on labeled datasets.

## 🤝 Contributors
- **👨‍💻 Project Lead:** [Your Name]
- **👥 Team Members:** [Other Contributors]
- **🙏 Acknowledgments:** [Relevant Mentions]

## 📜 License
This project is licensed under the MIT License. See `LICENSE` for details.


