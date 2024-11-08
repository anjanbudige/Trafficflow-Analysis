import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import tempfile
import shutil
model = YOLO("yolo11l.pt")
rush_threshold = 10
relevant_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']

class_names = model.names

st.title("Traffic Flow Analysis")
st.subheader("Real-time Vehicle Counting and Traffic Flow Detection")
option = st.radio("Choose Input", ("Upload Video", "Use Live Camera"))

def process_frame(frame):
    results = model(frame)

    car_count = 0
    truck_count = 0
    bus_count = 0
    motorcycle_count = 0
    bicycle_count = 0

    for result in results:
        boxes = result.boxes
        for det in boxes:
            class_id = int(det.cls)
            class_name = class_names[class_id]

            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)  # Coordinates for the box

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if class_name == 'car':
                car_count += 1
            elif class_name == 'truck':
                truck_count += 1
            elif class_name == 'bus':
                bus_count += 1
            elif class_name == 'motorcycle':
                motorcycle_count += 1
            elif class_name == 'bicycle':
                bicycle_count += 1

    total_vehicles = car_count + truck_count + bus_count + motorcycle_count + bicycle_count

    traffic_status = "Rush" if total_vehicles >= rush_threshold else "No Rush"

    cv2.putText(frame, f"Cars: {car_count}, Trucks: {truck_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Buses: {bus_count}, Motorcycles: {motorcycle_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Bicycles: {bicycle_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Traffic Flow: {traffic_status}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

if option == "Upload Video":
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name

        cap = cv2.VideoCapture(temp_file_path)
        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_frame(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        cap.release()

        if temp_file_path:
            shutil.remove(temp_file_path)

    else:
        st.warning("Please upload a video file to start traffic analysis.")

if option == "Use Live Camera":
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    stop_button = st.button("Stop Camera Feed", key="stop_camera_button")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Unable to access the camera.")
            break
        frame = process_frame(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        if stop_button:
            break
    cap.release()


# Add credits at the bottom of the page
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Designed by Anjan Budige</p>", unsafe_allow_html=True)
