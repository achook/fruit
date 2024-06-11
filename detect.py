import cv2
import numpy as np

# Choose the output directory to save the detected apples
OUTPUT_DIR = "output"

# Choose the video files to process, must be an array
VIDEO_PATHS = ["input/video_1.mp4", "input/video_2.mp4", "input/video_3.mp4", "input/video_4.mp4"]

# Set PREVIEW to True to display the detected apples in a window
PREVIEW = True

# Set SKIP_FRAMES to 1 to process every frame or higher to read every nth frame
SKIP_FRAMES = 1

def merge_overlapping_boxes(boxes):
    if not boxes:
        return []

    merged_boxes = []
    boxes = sorted(boxes, key=lambda b: b[0])

    while boxes:
        a = boxes.pop(0)
        merged = False
        for i in range(len(merged_boxes)):
            b = merged_boxes[i]
            if (a[0] <= b[0] + b[2] and a[0] + a[2] >= b[0] and
                a[1] <= b[1] + b[3] and a[1] + a[3] >= b[1]):
                # Merge boxes
                new_x = min(a[0], b[0])
                new_y = min(a[1], b[1])
                new_w = max(a[0] + a[2], b[0] + b[2]) - new_x
                new_h = max(a[1] + a[3], b[1] + b[3]) - new_y
                merged_boxes[i] = (new_x, new_y, new_w, new_h)
                merged = True
                break
        if not merged:
            merged_boxes.append(a)
    
    return merged_boxes

for video_idx, video_path in enumerate(VIDEO_PATHS):
    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Read every SKIP_FRAMES frames
    # Read until video is completed
    while cap.isOpened():
        for _ in range(SKIP_FRAMES - 1):
            cap.read()
            print("a")

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Display the resulting frame
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define the range of yellow and red color in HSV
            # TODO: Adjust the lower and upper values to detect the apples
            lower_yellow = np.array([19, 99, 110])
            upper_yellow = np.array([29, 200, 255])

            lower_red1 = np.array([170, 100, 86])
            upper_red1 = np.array([180, 245, 244])
            # lower_red2 = np.array([160, 100, 100])
            # upper_red2 = np.array([180, 255, 255])

            # Create a mask that isolates the yellow regions
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            # mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            # mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            mask_all = cv2.bitwise_or(mask_yellow, mask_red1)

            # Find contours in the mask
            contours_all, _ = cv2.findContours(mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_red, _ = cv2.findContours(mask_red1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            MIN_WIDTH = 4
            MIN_HEIGHT = 4
            padding = 7


            def check_image_dimension(w, h):
                # aspect_ratio = w / float(h)
                # return 0.5 <= aspect_ratio <= 1.5 and w > MIN_WIDTH and h > MIN_HEIGHT
                return w > MIN_WIDTH and h > MIN_HEIGHT
            
            bounding_boxes_yellow = []
            bounding_boxes_red = []


            # Loop over the contours
            for i, contour in enumerate(contours_yellow):
                # Get the bounding box for each contour
                x, y, w, h = cv2.boundingRect(contour)
                if check_image_dimension(w, h):
                    x_padded = max(0, x - padding)
                    y_padded = max(0, y - padding)
                    w_padded = min(frame.shape[1], x + w + padding) - x_padded
                    h_padded = min(frame.shape[0], y + h + padding) - y_padded

                    # Crop the apple from the original image
                    yellow_apple = frame[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded]

                    # Save the cropped apple image
                    cv2.imwrite(f'{OUTPUT_DIR}/yellow/unlabeled/apple_{video_idx+1}_{i}.jpeg', yellow_apple)

                    bounding_boxes_yellow.append((x_padded, y_padded, w_padded, h_padded)) 

            for i, contour in enumerate(contours_red):
                # Get the bounding box for each contour
                x, y, w, h = cv2.boundingRect(contour)

                # Crop the apple from the original image
                red_apple = frame[y:y + h, x:x + w]

                if check_image_dimension(w, h):
                    x_padded = max(0, x - padding)
                    y_padded = max(0, y - padding)
                    w_padded = min(frame.shape[1], x + w + padding) - x_padded
                    h_padded = min(frame.shape[0], y + h + padding) - y_padded

                    red_apple = frame[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded]
                    cv2.imwrite(f'{OUTPUT_DIR}/red/unlabeled/apple_{video_idx+1}_{i}.jpeg', red_apple)
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
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        else:
            break

    cap.release()
cv2.destroyAllWindows()    