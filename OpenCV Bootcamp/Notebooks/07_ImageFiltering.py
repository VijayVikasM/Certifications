import cv2
import sys
import numpy

# Constants for filter modes
PREVIEW = 0  # Preview Mode
BLUR = 1  # Blurring Filter
FEATURES = 2  # Corner Feature Detector
CANNY = 3  # Canny Edge Detector

# Parameters for corner detection
feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=15, blockSize=9)

# Default video source
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

# Initial state
image_filter = PREVIEW
alive = True

# Create a window
win_name = "Camera Filters"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
result = None

# Open the video source
source = cv2.VideoCapture(s)

while alive:
    # Capture a frame
    has_frame, frame = source.read()
    if not has_frame:
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)

    # Apply filters based on the current mode
    if image_filter == PREVIEW:
        result = frame
    elif image_filter == CANNY:
        result = cv2.Canny(frame, 80, 150)
    elif image_filter == BLUR:
        result = cv2.blur(frame, (13, 13))
    elif image_filter == FEATURES:
        result = frame.copy()  # Create a copy to draw on
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
        if corners is not None:
            for x, y in numpy.float32(corners).reshape(-1, 2):
                cv2.circle(result, (int(x), int(y)), 10, (0, 255, 0), 1)

    # Display the frame
    cv2.imshow(win_name, result)

    # Handle key presses
    key = cv2.waitKey(1)
    if key == ord("Q") or key == ord("q") or key == 27:  # Exit on 'Q', 'q', or Esc
        alive = False
    elif key == ord("C") or key == ord("c"):  # Canny filter
        image_filter = CANNY
    elif key == ord("B") or key == ord("b"):  # Blur filter
        image_filter = BLUR
    elif key == ord("F") or key == ord("f"):  # Feature detection filter
        image_filter = FEATURES
    elif key == ord("P") or key == ord("p"):  # Preview mode
        image_filter = PREVIEW

# Release resources
source.release()
cv2.destroyWindow(win_name)
