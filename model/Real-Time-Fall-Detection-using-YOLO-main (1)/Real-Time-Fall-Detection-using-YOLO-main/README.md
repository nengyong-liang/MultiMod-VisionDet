# Efficient Vision Based Fall Detection for Elderly Surveillance Using YOLOv11

**Official Code Repository for the IEEE Published Research** 📄 **[Read the Full Paper on IEEE Xplore](https://ieeexplore.ieee.org/document/11412994)** This project implements a real-time, lightweight fall detection system using the YOLOv11 deep learning framework. Designed specifically for elderly surveillance, the system accurately detects fall events from video streams in real-time, making it highly applicable for deployment in elderly care facilities, hospitals, and assisted living environments.

---

## Table of Contents

* [Key Contributions](https://www.google.com/search?q=%23key-contributions)
* [Project Architecture](https://www.google.com/search?q=%23project-architecture)
* [Datasets](https://www.google.com/search?q=%23datasets)
* [Model Details](https://www.google.com/search?q=%23model-details)
* [Installation](https://www.google.com/search?q=%23installation)
* [Usage](https://www.google.com/search?q=%23usage)
* [Training & Evaluation](https://www.google.com/search?q=%23training--evaluation)
* [Citation](https://www.google.com/search?q=%23citation)
* [Acknowledgements](https://www.google.com/search?q=%23acknowledgements)

---

## Key Contributions

Our research and implementation focus on optimizing state-of-the-art object detection for real-world edge deployment:

* **High Accuracy & Speed:** Fine-tuned YOLOv11 nano architecture on the LE2I dataset, achieving **95% accuracy** at a real-time processing speed of **25 FPS**.
* **False-Positive Reduction:** Implemented custom post-processing logic to filter out transient detections (e.g., bending, sitting rapidly), ensuring robust performance and minimizing false alarms.
* **Edge-Ready:** The lightweight nature of the model allows for scalable deployment on localized hardware, preserving privacy in sensitive healthcare settings.

---

## Project Architecture

The system is composed of the following components:

* **Video Input Module**: Captures real-time video feed from cameras.
* **Preprocessing Pipeline**: Resizes and normalizes video frames.
* **YOLO-based Object Detection**: Uses YOLOv11 to detect human figures and localize them with bounding boxes.
* **Fall Classification Logic**: Applies transient filtering and confidence scoring to distinguish between genuine fall events and normal daily activities.
* **Alert/Logging Module**: Records detected events and triggers notifications.

### Workflow Diagram

```text
+--------------------+       +-------------------+      +----------------------+
| Video Input Module | ----> | Preprocessing &   | ---> | YOLOv11 Object       |
| (Cameras/Feed)     |       | Data Augmentation |      | Detection            |
+--------------------+       +-------------------+      +----------------------+
                                                               |
                                                               v
                                         +----------------------------+
                                         | Fall Classification Logic  |
                                         | (Transient Filtering,      |
                                         | Confidence Scoring)        |
                                         +----------------------------+
                                                               |
                                                               v
                                        +-----------------------------+
                                        | Attendance/Alert Logging    |
                                        | (Database, Alerts)          |
                                        +-----------------------------+

```

---

## Datasets

### LE2I Dataset

* **Description**: The LE2I dataset is specifically designed for fall detection. It contains videos of real-life fall scenarios recorded from multiple angles, providing annotated data for both fall and non-fall activities.
* **Usage**: We fine-tuned YOLOv11 using this dataset to capture the variability of fall events under different lighting and environmental conditions.
* **Link**: [LE2I Dataset](https://universe.roboflow.com/le2iahlam/le2i-ahlam/model/1)

---

## Model Details

The YOLOv11 model used in this project is fine-tuned to detect falls with high recall and precision. Key training optimizations include:

* **Box Loss**: Optimizes the bounding box regression by comparing predicted boxes with ground truth using CIoU/GIoU metrics.
* **Class Loss**: Ensures the correct classification of detected objects using Binary Cross-Entropy.
* **Distribution Focal Loss (DFL)**: Refines localization by predicting a probability distribution over possible bounding box coordinates.

---

## Installation

### Prerequisites

* Python 3.7+
* PyTorch (version compatible with your CUDA setup)
* OpenCV
* Ultralytics YOLO

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/SyedBurhanAhmed/Real-Time-Fall-Detection-using-YOLO.git
cd Real-Time-Fall-Detection-using-YOLO

```


2. **Create a virtual environment and install dependencies:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```


3. **Download Pre-trained Weights:**
* Download the fine-tuned YOLOv11 weights (link provided in releases/repository).
* Place the weights file in the `./weights/` directory.



---

## Usage

### Running the Detection System

1. **Real-time Webcam Detection:**
```bash
python detect.py --source 0 --weights ./weights/best.pt --conf 0.4

```


2. **Batch Processing of Video Files:**
```bash
python detect.py --source path/to/video.mp4 --weights ./weights/best.pt --conf 0.4

```



---

## Training & Evaluation

To reproduce the results or fine-tune the model on your own dataset:

1. **Run Training:**
```bash
python train.py --data data.yaml --weights yolo11n.pt --epochs 50 --batch-size 16

```


2. **Run Evaluation:**
```bash
python val.py --data data.yaml --weights runs/train/exp/weights/best.pt

```



---

## Citation

If you find this code or our research helpful in your work, please cite our paper:

```bibtex
@INPROCEEDINGS{11412994,
  author={Ahmed, Syed Burhan and Habib, Shaista and Naveed, Khadija and Shakeel, Anousha and Rauf, Maira},
  booktitle={2025 6th International Conference on Innovative Computing (ICIC)}, 
  title={Efficient Vision Based Fall Detection For Elderly Surveillance Using YOLOv11}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Deep learning;Training;Accuracy;Computational modeling;Image edge detection;Video surveillance;Real-time systems;Fall detection;Older adults;Videos;deep learning;fall detection;human activity recognition;pre-trained models;real-time detection;video surveillance},
  doi={10.1109/ICIC68258.2025.11412994}
}


```



---

## Acknowledgements

We acknowledge the contributions of the research community in advancing object detection technologies and thank the maintainers of the LE2I dataset and the Ultralytics YOLO framework.

