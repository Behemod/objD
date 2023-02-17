import cv2
import numpy as np

# Load pre-trained Mask R-CNN model
net = cv2.dnn.readNetFromTensorflow('ImageAI/mytest/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb')

# Read input image
image = cv2.imread('ImageAI/mytest/1021801.jpg')

# Prepare input blob for neural network
blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)

# Set input and output layers
net.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False))
output_layer_names = ['detection_boxes', 'detection_scores', 'detection_classes', 'num_detections']
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Forward pass through neural network to get output
outputs = net.forward(output_layer_names)

# Extract segmentation mask from neural network output
mask = outputs[:, 1, :, :]

# Threshold mask to get binary segmentation mask
mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)[1]

# Invert mask to get background instead of object
mask = cv2.bitwise_not(mask)

# Apply mask to input image to remove background
result = cv2.bitwise_and(image, image, mask=mask)

# Display result
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()