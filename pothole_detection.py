import cv2
import numpy as np

# Load road video
cap = cv2.VideoCapture(
    r"C:\Users\SHAIK AAMEENA BBA\Downloads\anjum\pothole_road.mp4"
)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize
    frame = cv2.resize(frame, (900, 600))
    output = frame.copy()

    height, width = frame.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel X and Y gradients
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)

    # Gradient magnitude
    gradient = cv2.magnitude(gx, gy)

    # Normalize
    gradient = cv2.normalize(
        gradient,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    gradient = gradient.astype(np.uint8)

    # --------------------------------
    # ROAD REGION
    # --------------------------------

    mask = np.zeros((height, width), dtype=np.uint8)

    points = np.array([
        [0, height],
        [width, height],
        [width, int(height * 0.35)],
        [0, int(height * 0.35)]
    ], np.int32)

    cv2.fillPoly(mask, [points], 255)

    # Apply road mask
    road_gradient = cv2.bitwise_and(
        gradient,
        gradient,
        mask=mask
    )

    # --------------------------------
    # THRESHOLD
    # --------------------------------

    _, binary = cv2.threshold(
        road_gradient,
        50,
        255,
        cv2.THRESH_BINARY
    )

    # --------------------------------
    # CLOSE GAPS
    # --------------------------------

    kernel = np.ones((3, 3), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # --------------------------------
    # FIND CONTOURS
    # --------------------------------

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    pothole_count = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore very tiny edges
        if area < 300:
            continue

        # Ignore extremely large regions
        if area > 50000:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Ignore very small boxes
        if w < 25 or h < 15:
            continue

        # Aspect ratio
        ratio = w / float(h)

        if ratio < 0.25 or ratio > 5:
            continue

        # --------------------------------
        # POTENTIAL POTHOLE
        # --------------------------------

        pothole_count += 1

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        cv2.putText(
            output,
            "Potential Pothole",
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

    # Display count
    cv2.putText(
        output,
        "Potential Potholes: " + str(pothole_count),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    # Show windows
    cv2.imshow("Original Road Video", frame)
    cv2.imshow("Gradient Edge Map", road_gradient)
    cv2.imshow("Pothole Detection", output)

    # Press Q to quit
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()