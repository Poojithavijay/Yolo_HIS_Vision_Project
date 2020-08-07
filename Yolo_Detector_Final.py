import numpy as np
import argparse
import imutils
import time
import cv2
import os
from collections import Counter


# load the COCO class labels our YOLO model was trained on
labelsPath = r"D:\yolo-object-detection\yolo-coco\coco.names"
#labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = r"D:\yolo-object-detection\yolo-coco\yolov3.weights"
print("Weights Loaded")
configPath = r"D:\yolo-object-detection\yolo-coco\yolov3.cfg"
print("Model Config Loaded")

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(r"D:\yolo-object-detection\videos\overpass.mp4") 
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("Total frames in the video: {}".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []
	freq = []
	var = []
	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > 0.5:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
	#freq = [boxes.count(i) for i in boxes]


	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		var1 = 0
		var = []
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			cv2.line(frame, (415, 361), (615, 361), (0, 0, 0xFF), 3)
			cv2.line(frame, (670, 361), (843, 361), (0, 0, 0xFF), 3)
			freq1 = [j for j in classIDs]
			#freq = [[LABELS[classIDs[x]], classIDs.count(x)] for x in set(classIDs)]
			freq = dict([LABELS[x], classIDs.count(x)] for x in set(classIDs))
			print("Class:", freq, freq1)
			freq = str(freq)[1:-1]
			text1 = "{}".format(freq)
			cv2.putText(frame, text1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 2)
			cv2.putText(frame,"IN: ",(363, 355),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0xFF), 2)
			cv2.putText(frame, "OUT: ", (850, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0xFF), 2)
			#var1 = var1 + 1 if y > 350 else var1
			
			"""
			if y > 349 and y < 352:
				var1 = var1 + 1
				f = open(r"D:\yolo-object-detection\iter.txt", "a")
				f.write(str(var1))
				f.close()
				f = open(r"D:\yolo-object-detection\iter.txt", "r")
				var1 = f.read()
				f.close()
				var1 = sum(int(var1))
				text2 = "{}".format(var1)
				cv2.putText(frame,text2,(975, 350),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 0xFF),2)
				if y != 349 and y < 349:
					try:
						f = open(r"D:\yolo-object-detection\iter.txt", "r")
						var1 = f.read()
						f.close()
						var1 = sum(int(var1))
						text2 = "{}".format(var1)
						cv2.putText(frame,text2,(975, 350),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 0xFF),2)
					except:
						var1 = 0
			else:
				text2 = "{}".format(var1)
				cv2.putText(frame,text2,(975, 350),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 0xFF),2)
			"""
			var = 1
			
			#Detect Vehicles in Lane 1 (IN)
			if (x > 400 and x < 620 and y > 360 and y < 362.5):
				cv2.line(frame, (415, 361), (615, 361), (0, 0xFF, 0), 7)
				f = open("D:\yolo-object-detection\In.txt", "a")
				f.write("%d\n" %var)
				print("IN Dump Val:{}".format(var))
				f.close()
				#f = open(r"D:\yolo-object-detection\iter.txt", "r")
				#var1 = f.read()
				var1 = sum([int(s.strip()) for s in open("D:\yolo-object-detection\In.txt", "r").readlines()])
				print("IN Read Val:{}".format(var1))
				text2 = "{}".format(var1)
				cv2.putText(frame,text2,(390, 355),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0xFF), 2)
				f.close()

			else:
				try:
					var1 = sum([int(s.strip()) for s in open("D:\yolo-object-detection\In.txt", "r").readlines()])
					text2 = "{}".format(var1)
					cv2.putText(frame,text2,(390, 355),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 0xFF), 2)
					f.close()
				except:
					var1 = 0
					text2 = "{}".format(var1)
					cv2.putText(frame,text2,(390, 355),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0xFF), 2)

			
			#Detect Vehicles in Lane 2 (OUT)
			if (x > 670 and x < 845 and y > 360 and y < 362.5):
				cv2.line(frame, (670, 361), (843, 361), (0, 0xFF, 0), 7)
				f = open("D:\yolo-object-detection\Out.txt", "a")
				f.write("%d\n" %var)
				print("OUT Dump Val:{}".format(var))
				f.close()
				#f = open(r"D:\yolo-object-detection\iter.txt", "r")
				#var1 = f.read()
				var1 = sum([int(s.strip()) for s in open("D:\yolo-object-detection\Out.txt", "r").readlines()])
				print("OUT Read Val:{}".format(var1))
				text2 = "{}".format(var1)
				cv2.putText(frame,text2,(895, 355),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0xFF), 2)
				f.close()

			else:
				try:
					var1 = sum([int(s.strip()) for s in open("D:\yolo-object-detection\Out.txt", "r").readlines()])
					text2 = "{}".format(var1)
					cv2.putText(frame,text2,(895, 355),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 0xFF), 2)
					f.close()
				except:
					var1 = 0
					text2 = "{}".format(var1)
					cv2.putText(frame,text2,(895, 355),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0xFF), 2)
				
				
			#var.append(1) if y > 350 else 0
			

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(r"D:\yolo-object-detection\output\Test.avi", fourcc, 30, (frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
