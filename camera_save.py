# 拍照存储
import cv2
import time
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()# 读取摄像头画面
    cv2.imshow("frame",frame)
    key = cv2.waitKey(1)
    if key == 27:  # 按esc键退出
        print('esc break...')
        cap.release()
        cv2.destroyAllWindows()
        break

    if key == ord(' '):  # 按空格键保存
        # 保存一张图像
        filename = "config/" + str(time.time())[:10] + ".jpg"
        cv2.imwrite(filename, frame)
