[Industrial-Fastener-Sorting-AI.ipynb](https://github.com/user-attachments/files/29573916/Industrial-Fastener-Sorting-AI.ipynb)
# Real-Time Industrial Fastener Sorting AI

An Industry 4.0 computer vision application built with **YOLOv8** to automate quality control and object localization on manufacturing production lines. This deep learning system is custom-trained to instantly detect and classify core mechanical fasteners (Nuts, Bolts, and Washers) in real-time video streams.

---

##  Key Features
* **Real-time Detection:** Powered by a high-speed YOLOv8 Nano architecture optimized for real-time inference.
* **Custom Dataset Training:** Leveraged transfer learning on a specialized mechanical components dataset using custom bounding box labels.
* **GPU Accelerated Pipeline:** Trained using a cloud-based T4 GPU infrastructure to minimize model loss and maximize precision metrics.

---

## Tech Stack & Libraries
* **Language:** Python
* **Deep Learning Framework:** PyTorch & Ultralytics YOLOv8
* **Computer Vision & Graphics:** OpenCV, Matplotlib
* **Data Pipelines:** Roboflow API

---

## 
System Workflow & Implementation

### 1. Model Training
The model uses transfer learning from pre-trained COCO weights (`yolov8n.pt`) and maps them to custom industrial fastener classes. Training was executed for 25 epochs using a 640px image resolution matrix to lock in high confidence scores.

```python
from ultralytics import YOLO

# Load and train the custom pipeline
model = YOLO('yolov8n.pt')
model.train(data='path_to_dataset/data.yaml', epochs=25, imgsz=640, device=0)
