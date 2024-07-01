## VERSION 1
# import cv2
# import numpy as np
# import imutils
# from imutils.video import VideoStream, FPS
# import time

# # Load pre-trained model and configuration file
# prototxt = "deploy.prototxt"
# model = "mobilenet_iter_73000.caffemodel"

# # Load the model
# net = cv2.dnn.readNetFromCaffe(prototxt, model)

# # Define the list of class labels MobileNet SSD was trained to detect
# CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
#            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
#            "sofa", "train", "tvmonitor","knife","phone","scissors"]

# # Initialize the video stream
# cap = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     if not ret:
#         break

#     # Grab the frame dimensions and convert it to a blob
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), (127.5, 127.5, 127.5))

#     # Pass the blob through the network and obtain the detections and predictions
#     net.setInput(blob)
#     detections = net.forward()

#     # Loop over the detections
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]

#         # Filter out weak detections by ensuring the confidence is greater than a threshold
#         if confidence > 0.2:
#             # Extract the index of the class label from the detections
#             idx = int(detections[0, 0, i, 1])

#             # Compute the (x, y)-coordinates of the bounding box for the object
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")

#             # Draw the bounding box around the detected object
#             label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
#             cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
#             y = startY - 15 if startY - 15 > 15 else startY + 15
#             cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Display the resulting frame
#     cv2.imshow("Frame", frame)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture object and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

##VERSION 2
# import cv2
# import numpy as np
# import imutils
# from imutils.video import VideoStream, FPS
# import time

# # Load pre-trained model and configuration file
# prototxt = "deploy.prototxt"
# model = "mobilenet_iter_73000.caffemodel"

# # Load the model
# net = cv2.dnn.readNetFromCaffe(prototxt, model)

# # Define the list of class labels MobileNet SSD was trained to detect
# CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
#            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
#            "sofa", "train", "tvmonitor", "knife", "phone", "scissors",
#            "apple", "banana", "orange", "carrot", "sandwich", "pizza",
#            "donut", "cake", "chair", "couch", "bed", "table", "plant",
#            "mouse", "keyboard", "laptop", "cell phone", "book", "clock"]

# # Initialize the video stream
# cap = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     if not ret:
#         break

#     # Grab the frame dimensions and convert it to a blob
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), (127.5, 127.5, 127.5))

#     # Pass the blob through the network and obtain the detections and predictions
#     net.setInput(blob)
#     detections = net.forward()

#     # Loop over the detections
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]

#         # Filter out weak detections by ensuring the confidence is greater than a threshold
#         if confidence > 0.2:
#             # Extract the index of the class label from the detections
#             idx = int(detections[0, 0, i, 1])

#             # Compute the (x, y)-coordinates of the bounding box for the object
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")

#             # Draw the bounding box around the detected object
#             label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
#             cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
#             y = startY - 15 if startY - 15 > 15 else startY + 15
#             cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Display the resulting frame
#     cv2.imshow("Frame", frame)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture object and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

##FINAL VER.
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Load a pretrained YOLOv8 model

# Initialize the video stream
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Loop over the detections
    for result in results:
        boxes = result.boxes  # Get the boxes from the result

        for box in boxes:
            # Extract the bounding box coordinates and confidence score
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            class_name = model.names[class_id]

            # Draw the bounding box and label
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}: {confidence:.2f}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Frame", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
