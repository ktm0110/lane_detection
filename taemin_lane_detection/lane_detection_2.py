import cv2
import numpy as np

def region_of_interest(img, vertices):
    """
    관심 영역 설정 함수
    """
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines):
    """
    검출된 선을 이미지에 그리는 함수
    """
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

# 카메라 열기
cap = cv2.VideoCapture('./data/3.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 전처리
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)

    # 관심 영역 설정
    height, width = frame.shape[:2]
    vertices = np.array([[ (0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(canny, vertices)

    # Hough 변환
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    # 검출된 선을 이미지에 그리기
    line_image = draw_lines(frame, lines)

    # 결과 이미지 출력
    cv2.imshow('frame', line_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()