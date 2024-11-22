import cv2
import numpy as np

class DualLaneDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # 기본 카메라 사용
        if not self.cap.isOpened(): raise Exception("카메라를 열 수 없습니다.")
        print("카메라 초기화 완료")

    def process_frame(self, frame):
        # 이미지 크기
        height, width, _ = frame.shape

        # ROI 설정 (하단 40%)
        roi_height = int(height * 0.4)
        roi = frame[-roi_height:, :]

        # 그레이스케일 및 이진화
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        # 히스토그램 기반 초기 위치 찾기
        histogram = np.sum(binary[binary.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        left_base_x = np.argmax(histogram[:midpoint])  # 왼쪽 차선
        right_base_x = np.argmax(histogram[midpoint:]) + midpoint  # 오른쪽 차선

        # 슬라이딩 윈도우 파라미터
        num_windows = 10
        window_height = roi_height // num_windows
        margin = 50
        minpix = 50

        # 슬라이딩 윈도우로 차선 검출
        left_current_x = left_base_x
        right_current_x = right_base_x
        left_centers = []
        right_centers = []

        for window in range(num_windows):
            # 윈도우 높이
            win_y_low = roi_height - (window + 1) * window_height
            win_y_high = roi_height - window * window_height

            # 왼쪽 윈도우
            left_win_x_low = max(0, left_current_x - margin)
            left_win_x_high = min(width, left_current_x + margin)
            left_window_pixels = binary[win_y_low:win_y_high, left_win_x_low:left_win_x_high]
            left_nonzero = np.nonzero(left_window_pixels)
            if len(left_nonzero[0]) > minpix:
                left_current_x = int(np.mean(left_nonzero[1]) + left_win_x_low)
            left_centers.append((left_current_x, (win_y_low + win_y_high) // 2))

            # 오른쪽 윈도우
            right_win_x_low = max(0, right_current_x - margin)
            right_win_x_high = min(width, right_current_x + margin)
            right_window_pixels = binary[win_y_low:win_y_high, right_win_x_low:right_win_x_high]
            right_nonzero = np.nonzero(right_window_pixels)
            if len(right_nonzero[0]) > minpix:
                right_current_x = int(np.mean(right_nonzero[1]) + right_win_x_low)
            right_centers.append((right_current_x, (win_y_low + win_y_high) // 2))

            # 윈도우 시각화
            cv2.rectangle(roi, (left_win_x_low, win_y_low), (left_win_x_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(roi, (right_win_x_low, win_y_low), (right_win_x_high, win_y_high), (0, 255, 0), 2)

        # 차선 연결
        for i in range(1, len(left_centers)):
            cv2.line(roi, left_centers[i - 1], left_centers[i], (255, 0, 0), 5)
        for i in range(1, len(right_centers)):
            cv2.line(roi, right_centers[i - 1], right_centers[i], (0, 0, 255), 5)

        # 결과 시각화
        cv2.imshow("Original Frame", frame)
        cv2.imshow("ROI with Sliding Windows", roi)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.process_frame(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = DualLaneDetector()
    detector.run()
