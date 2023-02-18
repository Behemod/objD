import cv2
import rembg

# Load input image
input_path = 'ImageAI/mytest/1021807.jpg'
input_img = cv2.imread(input_path)

# Remove image background
output_img = rembg.bg.remove(input_img)

# RGBA converting to BGR
img = cv2.cvtColor(output_img, cv2.COLOR_RGBA2BGR)

# Convert input image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply smoothing to grayscale image
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 2)

# Apply morphological opening to remove small holes
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Find contours in the thresholded image
contours = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

# Iterate over contours and filter out those with area less than a threshold
filtered_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 10000:
        filtered_contours.append(contour)

# Draw filtered contours on original image
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(input_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(input_img, "Foreground", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display output image with label
cv2.imshow('Output', input_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
