# 🛶 River Floating Trash Detection System

This project is a **Computer Vision** application built using **Deep Learning** for automatic detection of various types of floating trash in river environments. Leveraging the power of **Faster R-CNN**, a state-of-the-art object detection architecture, this solution identifies and localizes trash items in video frames with high precision.

---

## 📌 Key Features

- 🔍 **Object Detection** using **Faster R-CNN with Feature Pyramid Networks (FPN)**
- 🧠 Fine-tuned Faster R-CNN model specifically for floating trash detection on river scenes
- 📦 Trained on a curated **Kaggle dataset** containing annotated floating trash
- 🗂️ Supports 8 different trash categories:
  - `bottle`, `grass`, `branch`, `milk-box`, `plastic-bag`, `plastic-garbage`, `ball`, `leaf`
- 🎥 Performs **real-time inference** on videos and exports annotated outputs
- 💡 Easily configurable and extendable for new object classes or environments

---

## 📂 Dataset

Dataset: [River Floating Trash Datasets](https://www.kaggle.com/datasets/zhiaun/river-floating-trash-datasets)

- **Total images:** 2400 (split across train, val, test)
- **Annotations:** Available in both YOLO and Pascal VOC formats, converted to COCO JSON
- **Image resolution:** 416×416
- **Classes:** 8 categories as listed above
- The dataset captures realistic scenarios of trash in rivers, making it suitable for environmental monitoring tasks.

---

## 🧠 Model Architecture

This project uses **Faster R-CNN with a ResNet-50 backbone and FPN** (Feature Pyramid Network):

- Architecture: `Faster R-CNN R50-FPN`
- Framework: `Detectron2` (from Facebook AI Research)
- Training:
  - Fine-tuned on the Kaggle dataset
  - Run for 10,000 iterations 

---

## 🚀 Quick Start

### 🔧 Requirements

- Python 3.10
- PyTorch 2.x
- OpenCV
- Detectron2

> ⚠️ On Windows, ensure you have Visual C++ Build Tools installed.

### 📥 Clone the repo and install dependencies

git clone https://github.com/your-username/river-trash-detection.git
cd river-trash-detection

🧪 Run Inference on a Video

Ensure your trained model (RFT_model.pth) and input video (RFT1.mp4) are in the same directory.

python RFT.py


### This script will:

Load the trained Faster R-CNN model

Read input video

Run detection on each frame

Save the annotated result to RFT1_output.mp4


## 📈 Evaluation Summary

Overall mAP@[0.50:0.95]: 42.4%

AP@0.50: 70.6%

Best performing classes:

ball: 59.8%

bottle: 56.9%

plastic-bag: 56.2%

### Challenging classes:

grass: 19.1%

branch: 30.1%

This performance indicates strong detection for well-defined object shapes but room for improvement with smaller or camouflaged objects.


## 📌 Project Files

RFT.py: Main inference script for processing videos using the trained model

River_Trash_Detection.ipynb: Notebook for training, evaluation, and visualization

RFT_model.pth: Trained model weights (Faster R-CNN)

test_coco_annotations.json: Converted annotations in COCO format


## 📚 Learn More

Detectron2 Documentation

Faster R-CNN Paper (2015)

## 🌍 Environmental Impact
This system directly contributes to sustainable development goals (SDGs) by promoting cleaner waterways, supporting smart waste management, and reducing manual effort in environmental monitoring. When combined with drones or IoT-based systems, it can help revolutionize the way cities manage river pollution — ensuring a healthier ecosystem for future generations.


## 🔮 Future Scope
🚁 Drone-Based River Surveillance
Deploy model on UAVs equipped with cameras

Enable aerial monitoring of large or remote water bodies

Automate detection and location tracking of trash clusters

### 🌐 IoT-Enabled Smart River Monitoring
Integrate with edge devices (Raspberry Pi, Jetson Nano)

Use stationary smart cameras at bridges, buoys, etc.

Trigger automatic alerts for cleanup teams

### ☁️ Cloud-Based Monitoring Dashboard
Real-time analytics dashboard using cloud platforms

Trash heatmaps, trend visualization, and alert logging

Integration with city-level waste management systems


## 🙋‍♂️ Author
### Ankush Naik
