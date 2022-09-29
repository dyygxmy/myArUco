
import cv2
import numpy as np
# import time

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
id=47
sidePixels=50
markerImage = np.zeros((sidePixels,sidePixels),dtype=np.uint8) # 全0就是一块黑的
# markerImage = np.ones((size, size), dtype=np.uint8)  # 全1也是一块黑的
markerImage=cv2.aruco.drawMarker(dictionary,id,sidePixels,markerImage,1) #生成ArUco标记
# OpenCV中的aruco模块共有25个预定义的标记字典。
# 字典中的所有标记包含相同数量的块或位（4×4、5×5、6×6 或 7×7），
# 每个字典包含固定数量的标记（50、100、250 或 1000）
# drawMarker(dictionary, id, sidePixels, img=None, borderBits=None)
# id 从0~249的标记集合中选择具有给定 id 的标记，这里是33
# sidePixels 生成一个 200×200 像素的图像
# img 存储生成的标记的对象
# borderBits 厚度参数，它决定了应该将多少块作为边界添加到生成的二进制模式中
cv2.imshow("markerImage",markerImage)
cv2.imwrite("img_2d/"+str(sidePixels)+"_"+str(id)+".png",markerImage)
cv2.waitKey()
cv2.destroyAllWindows()