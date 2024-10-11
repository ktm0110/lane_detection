import numpy as np
import cv2

def region_selection(image):
    mask = np.zeros_like(image)
    ignore_mask_color = 255

    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Hough Transform
def hough_transform(image):
    rho = 1
    theta = np.pi / 180
    threshold = 20
    min_line_length = 20
    max_line_gap = 300
    return cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def lane_lines(image, lines):
    left_lines = []
    right_lines = []
    left_weights = []
    right_weights = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:
                left_lines.append((slope, y1 - slope * x1))
                left_weights.append(np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
            else:
                right_lines.append((slope, y1 - slope * x1))
                right_weights.append(np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))

    left_line = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_line = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    y1 = image.shape[0]
    y2 = int(y1 * 0.6)

    left_lane = None if left_line is None else ((int((y1 - left_line[1]) / left_line[0]), y1),
                                               (int((y2 - left_line[1]) / left_line[0]), y2))
    right_lane = None if right_line is None else ((int((y1 - right_line[1]) / right_line[0]), y1),
                                                 (int((y2 - right_line[1]) / right_line[0]), y2))

    return left_lane, right_lane

def frame_processor(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    region = region_selection(edges)
    hough = hough_transform(region)
    return draw_lane_lines(image, lane_lines(image, hough))

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        lane_image = frame_processor(frame)

        cv2.imshow("Lane Detection", lane_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
