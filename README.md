# AI-Based Sign Language Recognition System

## Overview

The AI-Based Sign Language Recognition System is a real-time gesture recognition application developed using Computer Vision, MediaPipe, OpenCV, Flask, and Deep Learning techniques. The system is designed to recognize hand gestures through a webcam and convert them into meaningful text output, helping bridge the communication gap between hearing-impaired individuals and the general public.

The application captures live video input, detects hand landmarks using MediaPipe, extracts spatial features from the hand, and classifies gestures using a trained neural network model. The recognized gesture is then displayed in real time through a user-friendly web interface.

---

## Features

- Real-time hand gesture recognition
- Hand landmark detection using MediaPipe
- Deep Learning-based gesture classification
- Webcam-based gesture input
- Real-time prediction display
- Voice output support using Text-to-Speech
- Interactive web interface using Flask
- Support for multiple predefined gestures
- Expandable architecture for adding new gestures

---

## Supported Gestures

- HELLO
- YES
- NO
- HATE
- STOP
- ARE YOU SERIOUS
- WAIT
- PEACE
- LOSER
- DISLIKE
- LOVE YOU
- ROCK
- THREE
- OK

---

## Technologies Used

### Programming Language
- Python

### Computer Vision
- OpenCV
- MediaPipe

### Deep Learning
- TensorFlow
- Keras

### Web Framework
- Flask

### Frontend
- HTML
- CSS
- JavaScript

### Additional Libraries
- NumPy
- pyttsx3

---

## System Architecture

1. Webcam captures live video frames.
2. MediaPipe detects hand landmarks.
3. Landmark coordinates are extracted as feature vectors.
4. Features are processed by the trained deep learning model.
5. The model predicts the corresponding gesture.
6. The predicted gesture is displayed on the interface.
7. Text-to-Speech converts recognized gestures into spoken output.

---

## Dataset Information

The dataset consists of manually collected hand gesture samples captured under controlled conditions. Each gesture is recorded from multiple angles and positions to improve recognition accuracy and model robustness.

Dataset characteristics:

- 14 gesture classes
- Approximately 150–200 samples per gesture
- Single-hand gesture recognition
- Landmark-based feature extraction
- Real-time data collection using webcam

---

## Project Workflow

Data Collection → Hand Detection → Landmark Extraction → Data Preprocessing → Model Training → Gesture Classification → Text Generation → Voice Output

---

## Hardware Requirements

- Laptop/Desktop
- Webcam
- Minimum 4 GB RAM
- Intel i3 Processor or above

---

## Software Requirements

- Python 3.x
- OpenCV
- MediaPipe
- TensorFlow
- Flask
- NumPy
- pyttsx3

---

## Future Enhancements

- LSTM-based sequence modeling
- Dynamic gesture recognition
- Mobile application support
- Cloud deployment
- Speech-to-text integration
- Improved gesture vocabulary
- Multi-hand recognition support

---

## Applications

- Sign language communication assistance
- Educational platforms
- Accessibility solutions
- Human-computer interaction
- Smart assistance systems

---

## Installation

```bash
pip install -r requirements.txt
