****HIS Computer Vision Project: Vehicle Detection & Tracking with Yolo V3 (608x608) & OpenCV****


**Steps to setup & run the project:**

1. Install Python (python 3 is recommended)
2. Install PIP (mandatory)
3. Open CMD inside the project deliverables directory and run (mandatory step): pip install -r Requirement.txt
4. Download Yolo v3 Weights (608x608) from below link. I can't copy them to gitlab because of file size restrictions!
	**Link:** https://drive.google.com/drive/folders/1jFs9NSD_kiRR7wzLuC6o-IzBjzq9h0jW?usp=sharing
5. Save the "yolov3.weights" (downloaded in the above step) file inside the "yolo-coco" folder within the project directory. (Mandatory)
6. Download a sample input video from the link mentioned: https://drive.google.com/file/d/1k9mTMGVxDpLqlqr4T7mci-viInS5Pe_M/view?usp=sharing
7. Save the "overpass.mp4" file inside the videos folder within the project directory. (Mandatory)
8. Run the object detector model inside the project deliverables directory: python HIS.py -i videos/overpass.mp4 -o output/final_output.avi


**Steps to run the project inside container:**


**Information:** The final project docker image is publicly hosted in the Docker Hub portal: https://hub.docker.com/r/poojiyengar5/computer_vision

  **For Beginner Users:**
  (Who just wants to run the Object Detector once and see the results - cannot modify or view the code, so don't use this command if you want to continue the project development in the future!)
	
**Command**: docker run --rm --name HIS_Container -ti poojiyengar5/computer_vision:latest python3 HIS.py -i videos/overpass.mp4 -o output/final_output.avi

	For Advanced Users/For further development or modification of the project code:
		Please follow the following steps to run the project:
			Step 1: Container Creation
			Command: docker run --name HIS_Container -ti poojiyengar5/computer_vision:latest

			Step 2: Start the container
			Command: docker start HIS_Container

			Step 3 : Log-in to the container
			Command: docker exec -ti HIS_Container /bin/bash

			Step 4: Run the Object Detector
			Command: python3 HIS.py -i videos/overpass.mp4 -o output/test.avi

			Step 5: Modify the Object Detector Code
			Command: vi HIS.py

			Step 6: Exit from the container
			Command: exit

			Step 7: Stop the container
			Command: docker stop HIS_Container

			Step 8: Remove the container (if needed to delete the container)
			Command: docker rm HIS_Container
		

**Implemented Tasks:** 

1. Experimented & contrasted different YOLO v3 configurations.
2. Optimized Object Detection & Classification for large video streams.
3. Implemented vehicle density counter at per-frame/video level.
4. Developed efficient frame-level per-class object counter.
5. Designed Event-based object counter to detect incoming & outgoing vehicles in different lanes.
6. Refactored and optimized project code. 
7. Envisioned an End-to-End system by implementing a real-time object detection progress bar.
8. Set-up, executed & analysed the model on Google Cloud Platform and verified the overall model accuracies/fps.
9. Sped-up OpenCV‘s video writer output by utilizing different video extensions.
10. Fixed FFmpeg dependency issues in Ubuntu Docker container.


**Future Works:**

1. Post object detection generate detailed vehicle information reports for further statistical analysis.
2. Exploit pre-trained YOLO models from TensorFlow/PyTorch instead of OpenCV.
3. Rather than downloading OpenCV binaries from PIP/APT, compile OpenCV source code manually to enable CUDA backend.


**Author: Poojitha Vijayanarasimha (1293146)**
