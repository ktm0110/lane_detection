import numpy as np
import cv2

# 카메라 설정
Width = 640
Height = 480
cap = cv2.VideoCapture(0)

# Sliding Window 파라미터
nwindows = 9
margin = 50
minpix = 50

# 차선 색상 이진화 임계값
white_threshold = (200, 200, 200)
yellow_threshold = ([20, 100, 100], [30, 255, 255])  # HSV 기준


def preprocess_image(frame):
    """
    입력 영상을 전처리합니다.
    """
    # ROI 설정
    roi = frame[frame.shape[0] // 2:, :]

    # 하얀색 필터링
    white_mask = cv2.inRange(roi, white_threshold, (255, 255, 255))

    # 노란색 필터링 (HSV)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array(yellow_threshold[0]), np.array(yellow_threshold[1]))

    # 두 결과 합치기
    binary_output = cv2.bitwise_or(white_mask, yellow_mask)

    # 디버깅용 필터링 결과 출력
    cv2.imshow("Binary Output", binary_output)
    return binary_output


def sliding_window(binary_warped):
    """
    Sliding Window 알고리즘으로 차선을 검출합니다.
    """
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = int(binary_warped.shape[0] // nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, out_img


def draw_lane(original_img, binary_img, left_fit, right_fit):
    ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    for i in range(binary_img.shape[0]):
        cv2.line(original_img, (int(left_fitx[i]), int(ploty[i])), (int(right_fitx[i]), int(ploty[i])), (0, 255, 0), 10)
    return original_img


def start():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        binary_img = preprocess_image(frame)
        left_fit, right_fit, out_img = sliding_window(binary_img)
        cv2.imshow("Sliding Window", out_img)
        lane_img = draw_lane(frame, binary_img, left_fit, right_fit)
        cv2.imshow("Lane Detection", lane_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start()