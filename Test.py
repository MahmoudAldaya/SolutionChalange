import cv2

# Initialize the video capture object to read from the default camera
cap = cv2.VideoCapture(0)

# Define the rectangle to track the cars
x, y, w, h = 200, 200, 200, 200

# Set the threshold number of cars to turn the light to green
car_threshold = 5

# Initialize the number of cars to zero
car_count = 0

while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()

    # Extract the region of interest (ROI) within the rectangle
    roi = frame[y:y+h, x:x+w]

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the grayscale image to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply a binary threshold to the blurred image to create a black and white image
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a rectangle around the region of interest
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # If the number of contours in the ROI is greater than or equal to the car threshold,
    # increment the car count and turn the light to green
    if len(contours) >= car_threshold:
        car_count += 1
        cv2.putText(frame, "Green Light", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Red Light", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw the car count on the frame
    cv2.putText(frame, f"Cars: {car_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
