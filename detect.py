import cv2
import numpy as np

VIDEO_PATH = "input/video_3.mp4"
OUTPUT_DIR = "output"

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Display the resulting frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range of yellow and red color in HSV
        # TODO: Adjust the lower and upper values to detect the apples
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Create a mask that isolates the yellow regions
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_all = cv2.bitwise_or(mask_yellow, mask_red)

        # Find contours in the mask
        contours_all, _ = cv2.findContours(mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop over the contours
        for i, contour in enumerate(contours_yellow):
            # Get the bounding box for each contour
            # TODO: Add some pixel padding to the bounding box
            # TODO: Check if the aspect ratio of the bounding box is close to 1
            x, y, w, h = cv2.boundingRect(contour)

            # Crop the apple from the original image
            yellow_apple = frame[y:y + h, x:x + w]

            # Save the cropped apple image
            cv2.imwrite(f'{OUTPUT_DIR}/yellow/apple_{i}.jpeg', yellow_apple)

            # Draw the bounding box on the original image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for i, contour in enumerate(contours_red):
            # Get the bounding box for each contour
            # TODO: Add some pixel padding to the bounding box
            # TODO: Check if the aspect ratio of the bounding box is close to 1
            x, y, w, h = cv2.boundingRect(contour)

            # Crop the apple from the original image
            red_apple = frame[y:y + h, x:x + w]

            # Save the cropped apple image
            cv2.imwrite(f'{OUTPUT_DIR}/red/apple_{i}.jpeg', red_apple)

            # Draw the bounding box on the original image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


        cv2.imshow('Detected Apples', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
