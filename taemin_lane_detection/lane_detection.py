import cv2
import numpy as np

src = np.float32([[200, 720], [590, 460], [700, 460], [1100, 720]])
dst = np.float32([[300, 720], [300, 0], [900, 0], [900, 720]])

M = cv2.getPerspectiveTransform(src, dst)
# Minv = cv2.getPerspectiveTransform(dst, src)


def draw_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image,
 (x1, y1), (x2, y2), (0, 255, 0), thickness=10)


    return cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)

import cv2
import numpy as np

def process_image(image):

    # grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny Edge Detection
    canny = cv2.Canny(blur, 50, 150)

    # ROI
    imshape = image.shape
    vertices = np.array([[ (0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(canny, vertices)

    # Hough 변환
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # merge image and line image
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0.0)

    return result

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    result = process_image(frame)

    cv2.imshow('frame', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()