# Traffic Flow Analysis

This project is a Streamlit application for real-time traffic flow analysis using YOLOv8 for vehicle detection and counting. Users can upload a video file or use a live camera feed to analyze traffic flow and get real-time statistics on vehicle counts.

## Features

- **Real-Time Analysis**: Detect and count vehicles in real-time using the YOLOv11 model.
- **Traffic Flow Status**: Determine traffic status ("Rush" or "No Rush") based on the number of detected vehicles.
- **Flexible Input**: Users can upload a video file or use their live camera feed for analysis.

## Prerequisites

Make sure you have the following installed:

- **Python 3.9+**
- **[YOLOv11 Model](https://github.com/ultralytics/ultralytics)**: Pretrained model file (`yolo11l.pt`)
- **Streamlit**: For web application interface
- **OpenCV**: For video processing

## Installation
### 1. Clone the repository

```bash
git clone https://github.com/anjanbudige/Trafficflow-Analysis.git
cd Trafficflow-Analysis
```
### 2. Set up a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
### 3. Install dependencies

```bash
streamlit
opencv-python
ultralytics
```

## Usage
### Run the Streamlit App

```bash
streamlit run app.py
```
