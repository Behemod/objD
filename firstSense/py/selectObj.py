import cv2
import numpy as np

# Read input image
image = cv2.imread('firstSense/res/1021801.jpg')

# Convert input image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply smoothing to grayscale image
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Apply morphological opening to remove small holes
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Find contours in the thresholded image
contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate over contours and filter out those with area less than a threshold
filtered_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 1000:
        filtered_contours.append(contour)

# Draw filtered contours on original image
for contour in filtered_contours:
    # Approximate contour to reduce the number of points
    approx = cv2.approxPolyDP(contour, 0.0001*cv2.arcLength(contour, True), True)
    cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)

# Draw filtered contours on original image
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, "Foreground", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display labeled image
cv2.imshow("Labeled Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
