import cv2
import numpy as np

class MultiLaneDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # 기본 카메라 사용
        if not self.cap.isOpened():
            raise Exception("카메라를 열 수 없습니다.")
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

        # 히스토그램 분석
        histogram = np.sum(binary[binary.shape[0] // 2:, :], axis=0)

        # 히스토그램 피크 찾기
        peaks = self.find_peaks(histogram, threshold=1000, distance=100)

        # 슬라이딩 윈도우 파라미터
        num_windows = 10
        window_height = roi_height // num_windows
        margin = 50
        minpix = 50

        all_lane_centers = []

        # 각 피크에 대해 슬라이딩 윈도우 적용
        for peak in peaks:
            current_x = peak
            lane_centers = []

            for window in range(num_windows):
                win_y_low = roi_height - (window + 1) * window_height
                win_y_high = roi_height - window * window_height
                win_x_low = max(0, current_x - margin)
                win_x_high = min(width, current_x + margin)

                window_pixels = binary[win_y_low:win_y_high, win_x_low:win_x_high]
                nonzero = np.nonzero(window_pixels)

                if len(nonzero[0]) > minpix:
                    current_x = int(np.mean(nonzero[1]) + win_x_low)

                lane_centers.append((current_x, (win_y_low + win_y_high) // 2))

                # 윈도우 시각화
                cv2.rectangle(roi, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)

            all_lane_centers.append(lane_centers)

        # 차선 시각화
        colors = [(255, 0, 0), (0, 255, 255), (0, 0, 255), (255, 255, 0)]
        for i, lane_centers in enumerate(all_lane_centers):
            color = colors[i % len(colors)]  # 차선별로 색상 변경
            for j in range(1, len(lane_centers)):
                cv2.line(roi, lane_centers[j - 1], lane_centers[j], color, 5)

        # 결과 시각화
        cv2.imshow("Original Frame", frame)
        cv2.imshow("ROI with Sliding Windows", roi)

    def find_peaks(self, histogram, threshold=1000, distance=100):
        """
        히스토그램의 피크를 찾아 반환.
        :param histogram: 히스토그램 배열
        :param threshold: 피크로 간주할 최소 값
        :param distance: 피크 간 최소 거리
        :return: 피크의 x 좌표 리스트
        """
        peaks = []
        for i in range(len(histogram)):
            if histogram[i] > threshold:
                if not peaks or (i - peaks[-1]) > distance:
                    peaks.append(i)
        return peaks

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
    detector = MultiLaneDetector()
    detector.run()
