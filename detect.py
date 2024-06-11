import cv2
import numpy as np

from utils import merge_overlapping_boxes, check_image_dimension

# Choose the output directory to save the detected apples
OUTPUT_DIR = "output"

# Choose the video files to process, must be an array
VIDEO_PATHS = ["input/video_1.mp4", "input/video_2.mp4", "input/video_4.mp4"]

# Set PREVIEW to True to display the detected apples in a window
PREVIEW = False

# Set SKIP_FRAMES to 1 to process every frame or higher to read every nth frame
SKIP_FRAMES = 7

# Set the padding around the detected apple
APPLE_PADDING = 7


for video_idx, video_path in enumerate(VIDEO_PATHS):
    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Keep indexes for the filenames
    red_apple_idx = 0
    yellow_apple_idx = 0

    # Read every SKIP_FRAMES frames
    # Read until video is completed
    while cap.isOpened():
        for _ in range(SKIP_FRAMES - 1):
            cap.read()

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Display the resulting frame
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define the range of yellow and red color in HSV
            lower_yellow = np.array([19, 99, 110])
            upper_yellow = np.array([29, 200, 255])
            lower_red = np.array([170, 100, 86])
            upper_red = np.array([180, 245, 244])

            # Create a mask that isolates the yellow regions
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_red = cv2.inRange(hsv, lower_red, upper_red)
            mask_all = cv2.bitwise_or(mask_yellow, mask_red)

            # Find contours in the mask
            contours_all, _ = cv2.findContours(mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            bounding_boxes_yellow, bounding_boxes_red = [], []

            # Loop over the contours
            for i, contour in enumerate(contours_yellow):
                # Get the bounding box for each contour
                x, y, w, h = cv2.boundingRect(contour)
                if check_image_dimension(w, h):
                    x_padded = max(0, x - APPLE_PADDING)
                    y_padded = max(0, y - APPLE_PADDING)
                    w_padded = min(frame.shape[1], x + w + APPLE_PADDING) - x_padded
                    h_padded = min(frame.shape[0], y + h + APPLE_PADDING) - y_padded

                    # Crop the apple from the original image
                    yellow_apple = frame[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded]

                    # Save the cropped apple image
                    cv2.imwrite(f'{OUTPUT_DIR}/yellow/unlabeled/apple_{video_idx+1}_{yellow_apple_idx}.jpeg', yellow_apple)
                    yellow_apple_idx += 1

                    bounding_boxes_yellow.append((x_padded, y_padded, w_padded, h_padded)) 


            for i, contour in enumerate(contours_red):
                # Get the bounding box for each contour
                x, y, w, h = cv2.boundingRect(contour)

                if check_image_dimension(w, h):
                    x_padded = max(0, x - APPLE_PADDING)
                    y_padded = max(0, y - APPLE_PADDING)
                    w_padded = min(frame.shape[1], x + w + APPLE_PADDING) - x_padded
                    h_padded = min(frame.shape[0], y + h + APPLE_PADDING) - y_padded

                    # Crop the apple from the original image
                    red_apple = frame[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded]

                    cv2.imwrite(f'{OUTPUT_DIR}/red/unlabeled/apple_{video_idx+1}_{red_apple_idx}.jpeg', red_apple)
                    red_apple_idx += 1

                    bounding_boxes_red.append((x_padded, y_padded, w_padded, h_padded))

            bounding_boxes_yellow = merge_overlapping_boxes(bounding_boxes_yellow)
            bounding_boxes_red = merge_overlapping_boxes(bounding_boxes_red)
                    
            # Draw the bounding boxes
            for x, y, w, h in bounding_boxes_yellow:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                
            for x, y, w, h in bounding_boxes_red:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if PREVIEW:
                cv2.imshow('Detected apples', frame)

                # Press Q on keyboard to exit
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break
        else:
            break

    cap.release()
cv2.destroyAllWindows()
