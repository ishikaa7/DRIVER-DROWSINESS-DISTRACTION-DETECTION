# Driver Drowsiness & Distraction Detection 🚗💤

This project uses Computer Vision and Deep Learning to detect signs of driver fatigue and distraction in real-time. By monitoring facial landmarks, we calculate the Eye Aspect Ratio (EAR) to determine if a driver’s eyes are closed for an unsafe duration.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.0-green.svg)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange.svg)](https://opencv.org/)

A real-time Computer Vision application designed to prevent road accidents by detecting driver drowsiness, fatigue (yawning), and distraction (head pose).

---

## 🌟 Key Features
* **Drowsiness Detection**: Monitors Eye Aspect Ratio (EAR) with an audio alert system.
* **Fatigue Tracking**: Monitors Mouth Aspect Ratio (MAR) to count cumulative yawns.
* **Distraction Alert**: Detects when a driver looks away from the road for >1.5 seconds.
* **Modular Design**: Separated configuration, utilities, and main execution for easy calibration.

---

## 📐 The Methodology

### 1. Eye Aspect Ratio (EAR)
We utilize 16 specific landmarks for each eye. The EAR is calculated by finding the ratio between the vertical and horizontal distances of the eyelids.


### 2. Mouth Aspect Ratio (MAR)
The system tracks inner lip landmarks to detect wide-opening mouth events that signify a yawn.


### 3. Head Pose Estimation (Yaw)
By calculating the ratio of distance between the nose tip and the lateral edges of the face mesh, we determine if the driver's head is turned away from the center of the road.


---

## 📂 Project Architecture
The project is structured to be "Configuration-First," allowing for quick calibration without altering core logic:

* **`main.py`**: The central loop handling camera input and UI rendering.
* **`utils.py`**: All geometric calculations (Euclidean Distance, EAR, MAR, Head Direction).
* **`config.py`**: A centralized file for landmark indices, FPS thresholds, and sound frequencies.

---

## 🛠️ Installation & Usage

### 1. Requirements
* Python 3.8+
* Webcam

### 2. Setup

**`Clone the repository`**
git clone https://github.com/yourusername/driver-safety-system.git
cd driver-safety-system

**`Create and activate virtual environment`**
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

**`Install Dependencies`**
pip install opencv-python mediapipe 

### 3. Calibration Settings
Adjust these in config.py to match your specific camera angle and lighting:

(Parameter)                      (Current Value )                          (Purpose)
EAR_THRESHOLD                     0.15                   Threshold for eye closure (calibrated for low-angle views).
MAR_THRESHOLD                     0.6                    Threshold for detecting a wide-open mouth (yawn).
EYE_FPS_THRESHOLD                 30                     Minimum frames of closure before the alarm triggers (~1 sec).
BEEP_FREQ                         200                    Frequency (Hz) of the alert tone (Low frequency).