import cv2
import matplotlib.pyplot as plt
import time
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

count = 0

while True:
    count += 1
    begin = time.time()
    ret, frame = cap.read()

    if not ret:
        print("无法读取视频流")
        # break

    height, width, _ = frame.shape
    line_y = height // 2
    cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 0), 2)

    # 转换 BGR 到 RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 显示图像
    plt.imshow(frame)
    plt.axis('off')
    plt.pause(0.000001)  # 暂停一下，以便更新图像
    end = time.time()
    print("FPS: ", 1 / (end - begin))
    if count % 10 == 0:
        plt.close()

cap.release()
plt.close()
