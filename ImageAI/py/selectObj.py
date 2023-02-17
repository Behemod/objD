import cv2

# Load input image
image = cv2.imread("ImageAI/mytest/10212e29-68f3-4e03-8c72-bd7e06f0ece1.jpeg")

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold image to obtain binary image
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Find contours in the binary image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Check if only one contour was found
if len(contours) == 1:
    # Draw bounding box around contour
    x, y, w, h = cv2.boundingRect(contours[0])
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, "Foreground", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display labeled image
cv2.imshow("Labeled Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
