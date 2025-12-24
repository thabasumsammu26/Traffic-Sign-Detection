🚦 Traffic Sign Detection Using CNN (Real-Time)
📌 Overview

This project implements a real-time traffic sign detection and recognition system using Computer Vision and Deep Learning.
It captures live video from a webcam, detects traffic signs based on color and shape, and classifies them using a trained Convolutional Neural Network (CNN).

The system can recognize common traffic signs such as Stop, Speed Limits, No Entry, and warning signs, and display the detected sign along with its confidence score in real time.

🎯 Objectives

Detect traffic signs from live video feed

Classify detected signs accurately using a CNN model

Display sign name and prediction confidence in real time

Reduce false detections using image preprocessing and contour filtering

🛠️ Technologies Used

Python

OpenCV – Video capture and image processing

NumPy – Numerical operations

TensorFlow / Keras – Deep learning model

HSV Color Space – Robust color-based detection

⚙️ System Workflow

Capture live video using webcam

Convert frames from BGR to HSV color space

Detect red and blue regions using HSV thresholding

Apply contour detection to locate potential traffic signs

Crop and preprocess detected regions

Classify signs using a trained CNN model

Display bounding box, sign label, and confidence score

🧠 Model Details

Model Type: Convolutional Neural Network (CNN)

Input Size: 32 × 32 grayscale images

Dataset: GTSRB (German Traffic Sign Recognition Benchmark)

Output: Traffic sign class with probability

📂 Project Structure
├── traffif_sign_model.h5   # Trained CNN model
├── main.py                 # Real-time detection script
├── README.md               # Project documentation

▶️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/traffic-sign-detection.git
cd traffic-sign-detection

2️⃣ Install Dependencies
pip install numpy opencv-python tensorflow

3️⃣ Run the Application
python main.py


Press q to stop the webcam and exit.

📸 Output

Live webcam feed

Detected traffic sign highlighted with a bounding box

Display of:

Sign name (e.g., Stop)

Confidence score (e.g., 97.8%)

⚠️ Challenges Faced

Lighting variations: Affected detection accuracy

Solution: Used HSV color space instead of RGB

False positives: Red/blue objects detected as signs

Solution: Applied contour area and shape filtering

Real-time performance: Slight lag during prediction

Solution: Reduced frame resolution and optimized preprocessing

🚀 Real-Time Applications

Advanced Driver Assistance Systems (ADAS)

Autonomous Vehicles

Smart Traffic Monitoring

Fleet Management Systems

Traffic Violation Detection

Road Infrastructure Analysis

🔮 Future Enhancements

Integrate YOLO for faster object detection

Improve performance under low-light conditions

Add voice alerts for detected signs

Deploy as a web or mobile application

👨‍💻 Author
Syed Sameeda Thabasum
B.Tech CSE 

Shaik Mahammad Iliyaz
B.Tech CSE | AI & Web Development Enthusiast
