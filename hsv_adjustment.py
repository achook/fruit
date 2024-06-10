import cv2
import numpy as np
import os



def filter_outliers(data, m=2):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    filtered_data = data[np.all(abs(data - mean) <= m * std, axis=1)]
    return np.min(filtered_data, axis=0), np.max(filtered_data, axis=0)

cropped_dir = './yellow_samples/'
images = os.listdir(cropped_dir)

hsv_values = []

# Loop over all images
for img_name in images:
    img_path = os.path.join(cropped_dir, img_name)
    img = cv2.imread(img_path)

    if img is not None:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_values.extend(hsv_img.reshape(-1, 3)) 

hsv_values = np.array(hsv_values)  

# Filter outliers using Standard Deviation
min_hsv_std, max_hsv_std = filter_outliers(hsv_values, m=2)

# Calculate percentiles for a less aggressive filter
min_hsv_percentile = np.percentile(hsv_values, 5, axis=0)
max_hsv_percentile = np.percentile(hsv_values, 95, axis=0)

print("HSV Limits using Standard Deviation Filter:")
print("Min HSV:", min_hsv_std)
print("Max HSV:", max_hsv_std)

print("HSV Limits using Percentile Filter:")
print("Min HSV:", min_hsv_percentile)
print("Max HSV:", max_hsv_percentile)


def nothing(x):
    pass


cv2.namedWindow('HSV Adjuster')

initial_lower_h = 19
initial_upper_h = 29
initial_lower_s = 99
initial_upper_s = 200
initial_lower_v = 110
initial_upper_v = 255

# Create trackbars 
cv2.createTrackbar('Lower H', 'HSV Adjuster', initial_lower_h, 180, nothing)
cv2.createTrackbar('Upper H', 'HSV Adjuster', initial_upper_h, 180, nothing)
cv2.createTrackbar('Lower S', 'HSV Adjuster', initial_lower_s, 255, nothing)
cv2.createTrackbar('Upper S', 'HSV Adjuster', initial_upper_s, 255, nothing)
cv2.createTrackbar('Lower V', 'HSV Adjuster', initial_lower_v, 255, nothing)
cv2.createTrackbar('Upper V', 'HSV Adjuster', initial_upper_v, 255, nothing)
cap = cv2.VideoCapture('input/video_3.mp4')
playback_delay = 1  # Delay 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current positions of the trackbars
    l_h = cv2.getTrackbarPos('Lower H','HSV Adjuster')
    u_h = cv2.getTrackbarPos('Upper H','HSV Adjuster')
    l_s = cv2.getTrackbarPos('Lower S','HSV Adjuster')
    u_s = cv2.getTrackbarPos('Upper S','HSV Adjuster')
    l_v = cv2.getTrackbarPos('Lower V','HSV Adjuster')
    u_v = cv2.getTrackbarPos('Upper V','HSV Adjuster')


    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    # Create masks
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)

    if cv2.waitKey(playback_delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

