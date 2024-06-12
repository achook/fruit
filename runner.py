import cv2
import numpy as np
from keras import models

from utils import merge_overlapping_boxes, check_image_dimension

# Choose the video files to process, must be an array
VIDEO_PATH = "input/video_3.mp4"

# Set the padding around the detected apple
APPLE_PADDING = 7

# The minimum confidence to consider an apple detected (0.0 - 1.0)
# The higher the value, the more confident the network has to be for the image to be considered as an apple
RED_APPLE_DETECT_MIN_CONFIDENCE = 0.5
YELLOW_APPLE_DETECT_MIN_CONFIDENCE = 0.5

NETWORK_INPUT_SHAPE = (50, 50, 3)

SHOW_UNSURE = False

# Load the models
model_red = models.load_model("models/red_apple_model.keras")
model_yellow = models.load_model("models/yellow_apple_model.keras")

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)

# Read until video is completed
while cap.isOpened():

    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Display the resulting frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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
        bounding_boxes_yellow_unsure, bounding_boxes_red_unsure = [], []

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

                # resize apple to 50x50 pixels, pad if the image is not square
                if w_padded > h_padded:
                    pad = (w_padded - h_padded) // 2
                    yellow_apple = cv2.copyMakeBorder(yellow_apple, pad, pad, 0, 0, cv2.BORDER_CONSTANT,
                                                      value=(0, 0, 0))
                elif h_padded > w_padded:
                    pad = (h_padded - w_padded) // 2
                    yellow_apple = cv2.copyMakeBorder(yellow_apple, 0, 0, pad, pad, cv2.BORDER_CONSTANT,
                                                      value=(0, 0, 0))

                yellow_apple = cv2.resize(yellow_apple, NETWORK_INPUT_SHAPE[:2])
                yellow_apple = np.expand_dims(yellow_apple, axis=0)

                # Predict the apple
                prediction = model_yellow.predict(yellow_apple, verbose=0)
                if prediction > YELLOW_APPLE_DETECT_MIN_CONFIDENCE:
                    bounding_boxes_yellow.append((x_padded, y_padded, w_padded, h_padded))
                else:
                    bounding_boxes_yellow_unsure.append((x_padded, y_padded, w_padded, h_padded))

        for i, contour in enumerate(contours_red):
            # Get the bounding box for each contour
            x, y, w, h = cv2.boundingRect(contour)


            if check_image_dimension(w, h):
                x_padded = max(0, x - APPLE_PADDING)
                y_padded = max(0, y - APPLE_PADDING)
                w_padded = min(frame.shape[1], x + w + APPLE_PADDING) - x_padded
                h_padded = min(frame.shape[0], y + h + APPLE_PADDING) - y_padded

                red_apple = frame[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded]

                if w_padded > h_padded:
                    pad = (w_padded - h_padded) // 2
                    red_apple = cv2.copyMakeBorder(red_apple, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                elif h_padded > w_padded:
                    pad = (h_padded - w_padded) // 2
                    red_apple = cv2.copyMakeBorder(red_apple, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

                red_apple = cv2.resize(red_apple, (50, 50))
                red_apple = np.expand_dims(red_apple, axis=0)

                # Predict the apple
                prediction = model_red.predict(red_apple, verbose=0)
                if prediction[0][0] > RED_APPLE_DETECT_MIN_CONFIDENCE:
                    bounding_boxes_red.append((x_padded, y_padded, w_padded, h_padded))
                else:
                    bounding_boxes_red_unsure.append((x_padded, y_padded, w_padded, h_padded))

        bounding_boxes_yellow_unsure = merge_overlapping_boxes(bounding_boxes_yellow_unsure)
        bounding_boxes_red_unsure = merge_overlapping_boxes(bounding_boxes_red_unsure)
        bounding_boxes_yellow = merge_overlapping_boxes(bounding_boxes_yellow)
        bounding_boxes_red = merge_overlapping_boxes(bounding_boxes_red)

        # Draw the bounding boxes
        for x, y, w, h in bounding_boxes_yellow:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        for x, y, w, h in bounding_boxes_red:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if SHOW_UNSURE:
            for x, y, w, h in bounding_boxes_yellow_unsure:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 128), 2)

            for x, y, w, h in bounding_boxes_red_unsure:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 128), 2)

        cv2.imshow('Detected apples', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
