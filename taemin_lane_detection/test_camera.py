import cv2

camera_indices = [0, 1]

for index in camera_indices:
    print(f"Testing camera with index {index}...")
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Camera index {index} could not be opened.")
        continue

    ret, frame = cap.read()
    if not ret:
        print(f"Camera index {index} could not read a frame.")
    else:
        print(f"Camera index {index} is working!")
        cv2.imshow(f"Camera {index}", frame)
        cv2.waitKey(2000)  # 2sec

    cap.release()
    cv2.destroyAllWindows()
