"""
拍照存储
主要是拍摄棋盘图片，准备标定用
"""
import cv2
import os
from GIGETest import getImageData_init,newsStreamSource,getSourceImages,clearCache

def collect_material():
    isGIGE = True
    if isGIGE:
        camera = getImageData_init()
        streamSource = newsStreamSource(camera)
    else:
        cap = cv2.VideoCapture(0)
    cout=100
    names=os.listdir("masterImg")
    for name in names:
        os.remove("masterImg/"+name)
    while True:
        if isGIGE:
            status,frame = getSourceImages(streamSource)
            if status == -1:
                continue
        else:
            ret, frame = cap.read()# 读取摄像头画面
        cv2.imshow("frame",frame)
        key = cv2.waitKey(1)
        if key == 27:  # 按esc键退出
            print('esc break...')
            if isGIGE:
                clearCache(streamSource, camera, True)
            else:
                cap.release()
            cv2.destroyAllWindows()
            break

        if key == ord(' '):  # 按空格键保存
            # 保存一张图像
            cout+=1
            filename = "masterImg/master" + str(cout)[1:] + ".jpg"
            cv2.imwrite(filename, frame)

if __name__ == "__main__":
    collect_material()