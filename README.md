HIS Computer Vision Project: Vehicle Detection & Tracking with Yolo V3 (608x608) & OpenCV

Download Yolo v3 output videos & model weight from below link. (I can't copy them to github because of size restrictions!)

Link: https://drive.google.com/drive/folders/1jFs9NSD_kiRR7wzLuC6o-IzBjzq9h0jW?usp=sharing
Download final Docker Container Image: "docker pull poojiyengar5/computer_vision:latest"

#Code Changes:
1. Changed the input method from argparse
2. Altered the video writer output to AVI for better processing speed
3. Implemented a frame-level total object counter to understand the density of objects at any point of time
4. Implemented frame-level class-wise object counter
5. Implemented ROI mechanism to detect vehicles crossing the ROI line
6. Implemented & Optimised IN and OUT counter to count vehicles in different lanes.


Further tasks:
1. Run the OpenCV model with CUDA backend
