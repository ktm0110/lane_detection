import numpy as np
import cv2, random, math, copy

# 기본 카메라 해상도 설정
Width = 640
Height = 480

# 카메라 캡처 설정
cap = cv2.VideoCapture(0)
window_title = 'camera'  # 카메라 창 이름

# 원근 변환 후 이미지의 크기 설정
warp_img_w = 320
warp_img_h = 240

# 슬라이딩 윈도우 설정
nwindows = 9  # 윈도우의 수
margin = 12  # 각 윈도우의 폭
minpix = 5  # 최소 픽셀 수 (차선을 따라 움직일 최소 픽셀 수)

# 차선 이진화 임계값
lane_bin_th = 145

# 원근 변환의 소스 좌표 설정 (도로 영역을 사다리꼴 모양으로 지정)
warp_src = np.array([
    [210, 297],  # 왼쪽 상단
    [42, 453],   # 왼쪽 하단
    [465, 297],  # 오른쪽 상단
    [630, 453]   # 오른쪽 하단
], dtype=np.float32)

# 원근 변환의 목적지 좌표 설정 (직사각형으로 평탄하게 변환될 위치)
warp_dist = np.array([
    [0, 0],
    [0, warp_img_h],
    [warp_img_w, 0],
    [warp_img_w, warp_img_h]
], dtype=np.float32)

# 카메라 보정 파라미터 (캘리브레이션 여부 확인)
calibrated = True
if calibrated:
    # 카메라 내부 파라미터 설정
    mtx = np.array([
        [422.037858, 0.0, 245.895397],
        [0.0, 435.589734, 163.625535],
        [0.0, 0.0, 1.0]
    ])
    # 렌즈 왜곡 계수 설정
    dist = np.array([-0.289296, 0.061035, 0.001786, 0.015238, 0.0])
    # 최적의 카메라 매트릭스 계산
    cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (Width, Height), 1, (Width, Height))


# 카메라 보정 함수 (렌즈 왜곡을 보정)
def calibrate_image(frame):
    global Width, Height
    global mtx, dist
    global cal_mtx, cal_roi

    # 왜곡 보정된 이미지 생성
    tf_image = cv2.undistort(frame, mtx, dist, None, cal_mtx)
    # 유효 영역만큼 이미지를 크롭
    x, y, w, h = cal_roi
    tf_image = tf_image[y:y + h, x:x + w]
    # 보정된 이미지를 원본 크기로 리사이즈
    return cv2.resize(tf_image, (Width, Height))


# 원근 변환을 적용하여 버드 아이 뷰 생성
def warp_image(img, src, dst, size):
    # 변환 매트릭스 M과 역변환 매트릭스 Minv 생성
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # 원근 변환 적용
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

    return warp_img, M, Minv


# 변환된 이미지에서 차선 검출 (슬라이딩 윈도우 방식)
def warp_process_image(img):
    global nwindows
    global margin
    global minpix
    global lane_bin_th

    # 블러링 및 HLS 색상 변환 후 이진화 처리
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, L, _ = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2HLS))
    _, lane = cv2.threshold(L, lane_bin_th, 255, cv2.THRESH_BINARY)

    # 히스토그램을 이용해 왼쪽과 오른쪽 차선의 기초 위치를 결정
    histogram = np.sum(lane[lane.shape[0] // 2:, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    leftx_current = np.argmax(histogram[:midpoint])
    rightx_current = np.argmax(histogram[midpoint:]) + midpoint

    # 슬라이딩 윈도우의 높이 설정
    window_height = int(lane.shape[0] / nwindows)
    nz = lane.nonzero()

    left_lane_inds = []
    right_lane_inds = []

    lx, ly, rx, ry = [], [], [], []

    # 차선 시각화를 위한 초기 이미지 설정
    out_img = np.dstack((lane, lane, lane)) * 255

    # 슬라이딩 윈도우 방식으로 차선을 추적
    for window in range(nwindows):
        # 각 윈도우의 세로 범위 설정
        win_yl = lane.shape[0] - (window + 1) * window_height
        win_yh = lane.shape[0] - window * window_height

        # 왼쪽과 오른쪽 윈도우의 가로 범위 설정
        win_xll = leftx_current - margin
        win_xlh = leftx_current + margin
        win_xrl = rightx_current - margin
        win_xrh = rightx_current + margin

        # 윈도우 범위를 시각적으로 표시
        cv2.rectangle(out_img, (win_xll, win_yl), (win_xlh, win_yh), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xrl, win_yl), (win_xrh, win_yh), (0, 255, 0), 2)

        # 각 윈도우 내에서 유효한 차선 픽셀 찾기
        good_left_inds = ((nz[0] >= win_yl) & (nz[0] < win_yh) & (nz[1] >= win_xll) & (nz[1] < win_xlh)).nonzero()[0]
        good_right_inds = ((nz[0] >= win_yl) & (nz[0] < win_yh) & (nz[1] >= win_xrl) & (nz[1] < win_xrh)).nonzero()[0]

        # 찾은 픽셀들을 리스트에 추가
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # 유효한 픽셀이 충분하면 다음 윈도우 위치 조정
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nz[1][good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nz[1][good_right_inds]))

        # 현재 윈도우 위치 저장
        lx.append(leftx_current)
        ly.append((win_yl + win_yh) / 2)
        rx.append(rightx_current)
        ry.append((win_yl + win_yh) / 2)

    # 리스트를 하나의 배열로 병합
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # 좌우 차선을 2차 함수로 피팅
    lfit = np.polyfit(np.array(ly), np.array(lx), 2)
    rfit = np.polyfit(np.array(ry), np.array(rx), 2)

    # 시각화 이미지에 차선 표시
    out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [0, 0, 255]
    out_img[nz[0][right_lane_inds], nz[1][right_lane_inds]] = [255, 0, 0]
    cv2.imshow("viewer", out_img)

    return lfit, rfit


# 차선을 시각적으로 그리는 함수
def draw_lane(image, warp_img, Minv, left_fit, right_fit):
    global Width, Height
    yMax = warp_img.shape[0]
    ploty = np.linspace(0, yMax - 1, yMax)
    color_warp = np.zeros_like(warp_img).astype(np.uint8)

    # 좌우 차선의 x 좌표 계산
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # 차선을 그릴 좌표 설정
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # 왼쪽과 오른쪽 차선을 색깔로 표시
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(0, 0, 255), thickness=10)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(255, 0, 0), thickness=10)

    # 차선 영역을 녹색으로 채우기
    color_warp = cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (Width, Height))

    return cv2.addWeighted(image, 1, newwarp, 0.3, 0)


# Warp 영역을 시각적으로 표시하는 함수
def draw_warp_area(image, src):
    pts = src.reshape((-1, 1, 2))
    cv2.polylines(image, [np.int32(pts)], isClosed=True, color=(0, 255, 0), thickness=2)
    return image


# 메인 루프 - 실시간 영상 처리
def start():
    global Width, Height, cap

    if not cap.isOpened():
        print("can't open")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("can't read")
            break

        # Warp 영역을 원본 이미지에 표시
        frame_with_warp_area = draw_warp_area(frame, warp_src)

        # 이미지 보정 및 Warp 적용
        image = calibrate_image(frame)
        warp_img, M, Minv = warp_image(image, warp_src, warp_dist, (warp_img_w, warp_img_h))
        left_fit, right_fit = warp_process_image(warp_img)
        lane_img = draw_lane(image, warp_img, Minv, left_fit, right_fit)

        # 창에 이미지 표시
        cv2.imshow("Warp Area", frame_with_warp_area)
        cv2.imshow(window_title, lane_img)

        # 종료 조건 설정
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# 프로그램 시작
if __name__ == '__main__':
    start()
