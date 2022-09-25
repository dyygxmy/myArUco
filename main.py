

import cv2 as cv
import numpy as np

def main(name):
    # 在下面的代码行中使用断点来调试脚本。
    # print(f'Hi, {name}')
    # dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
    # # markerImage = np.zeros((200,200),dtype=np.uint8) # 全0就是一块黑的
    # markerImage = np.ones((500, 500), dtype=np.uint8)  # 全1也是一块黑的
    # markerImage=cv.aruco.drawMarker(dictionary,33,200,markerImage,1) #生成ArUco标记
    # OpenCV中的aruco模块共有25个预定义的标记字典。
    # 字典中的所有标记包含相同数量的块或位（4×4、5×5、6×6 或 7×7），
    # 每个字典包含固定数量的标记（50、100、250 或 1000）
    #drawMarker(dictionary, id, sidePixels, img=None, borderBits=None)
    #id 从0~249的标记集合中选择具有给定 id 的标记，这里是33
    #sidePixels 生成一个 200×200 像素的图像
    #img 存储生成的标记的对象
    #borderBits 厚度参数，它决定了应该将多少块作为边界添加到生成的二进制模式中
    # cv.imshow("markerImage",markerImage)
    # cv.imwrite("test.png",markerImage)
    # cv.waitKey()
    # cv.destroyAllWindows()


    # 检测ArUco标记
    # frame = cv.imread("marker33.png",1)
    # # Load the dictionary that was used to generate the markers.
    # dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
    # # Initialize the detector parameters using default values
    # parameters = cv.aruco.DetectorParameters_create()
    # # Detect the markers in the image
    # markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    # 通过分析二维码确定marker 在opencv的ArUco模块中，主要通过detectMarkers()函数来完成，
    # 这个函数是整个模块中最重要的函数了，因为后续的函数处理几乎都依赖于该函数的检测结果
    #对于每个成功的标记检测，从左上角、右上角、右下角和左下角依次检测标记的四个角点。
    # 在 C++ 中，这 4 个检测到的角点存储为点向量，图像中的多个标记一起存储在点向量向量中。
    # 在 Python 中，它们存储为 Numpy 数组。
    #该detectMarkers功能用于检测和定位标记的角落。
    # 第一个参数是带有标记的场景图像。
    # 第二个参数是用于生成标记的字典。
    # 成功检测到的标记将存储在 中markerCorners，
    # 它们的 id 将存储在markerIds.
    # 之前初始化的 DetectorParameters 对象也作为参数传递。
    # 最后，被拒绝的候选者存储在rejectedCandidates
    #在场景中打印、切割和放置标记时，重要的是我们在标记的黑色边界周围保留一些白色边框，以便可以轻松检测到它们
    # print("markerCorners",markerCorners)
    # print("markerIds", markerIds)
    # print("rejectedCandidates", rejectedCandidates)



    #现实应用
    #我们使用新的场景图像角点作为源点（pts_src），并将我们捕获的图像中相框内的相应图片角点作为目标点（dst_src）。
    # OpenCV 函数findHomography计算源点和目标点之间的单应函数h 。然后使用单应矩阵来扭曲新图像以适合目标帧。
    # 扭曲的图像被屏蔽并复制到目标帧中。在视频的情况下，这个过程在每一帧上重复。

    im_back = cv.imread("back.png")
    im_base=cv.imread("base.png")
    pts_base=np.array([[0,0],[840,0],[840,560],[0,560]])
    pts_new=np.array([[419,150],[775,115],[782,520],[410,509]])
    # im_dstDraw = cv.polylines(im_src, [np.int32(pts_dst)],True,[255,0,0],thickness=4,lineType=cv.LINE_AA)
    #polylines(img, pts, isClosed, color, thickness=None, lineType=None, shift=None)
    # cv.imshow("im_dstDraw", im_dstDraw)

    # 在原图im_back上画一个多边形(直接对im_back生效，返回值也是生效后的im_back)
    cv.polylines(im_back, [np.int32(pts_new)], True, [0,0,255], 3, cv.LINE_AA) #红色画线，线粗3
    # cv.imshow("im_dstDraw", im_dstDraw)
    # cv.imshow("im_back", im_back)
    cv.imwrite("im_dstDraw.png",im_back)

    # 计算单应性矩阵 这个是重点
    # h1, status = cv.findHomography(pts_src, pts_dst) # pts_src矩阵映射成pts_dst矩阵
    # # 对图像进行透视变换，就是变形 把im_dst变形匹配im_src
    # im_out = cv.warpPerspective(im_dst, h1, (im_src.shape[1], im_src.shape[0]))
    # # im_out=cv.add(cv.multiply(im_min,1),cv.multiply(im_src,2))




    # pts_src矩阵映射成pts_dst矩阵
    h, status = cv.findHomography(pts_base,pts_new )

    # 将原图像变形后放到黑色背景图像(大小按im_back)上
    warped_image = cv.warpPerspective(im_base, h, (im_back.shape[1], im_back.shape[0]))
    cv.imwrite("warped_image.png",warped_image)

    # 准备一个与im_back一样大小表示区域的蒙版mask
    mask = np.zeros([im_back.shape[0], im_back.shape[1]], dtype=np.uint8)
    cv.imwrite("mask.png",mask)
    # 按扭曲的图像区域填充白色到mask上(直接对mask生效，返回值也是生效后的mask)
    cv.fillConvexPoly(mask, np.int32([pts_new]), (255, 255, 255), cv.LINE_AA)
    cv.imwrite("fill.png",mask)

    # 把mask填充白色的区域再抹掉一圈，避免从扭曲中复制边界效果
    element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mask = cv.erode(mask, element, iterations=3)
    cv.imwrite("mask2.png", mask)

    # 将蒙版复制到3个通道
    warped_image = warped_image.astype(float)
    mask3 = np.zeros_like(warped_image)
    for i in range(0, 3):
        mask3[:, :, i] = mask / 255
    cv.imwrite("mask3.png", mask3)

    #将蒙版扭曲图像复制到蒙版区域的原始帧中(其他地方填充黑色)
    warped_image_masked = cv.multiply(warped_image, mask3)
    cv.imwrite("warped_image_masked.png",warped_image_masked)
    #将原背景图对应mask黑色部分(其他地方填充黑色)
    frame_masked = cv.multiply(im_back.astype(float), 1 - mask3)
    cv.imwrite("frame_masked.png",frame_masked)
    #将两个图去掉黑色的部分，合在一起
    im_out = cv.add(warped_image_masked, frame_masked)
    cv.imwrite("im_out.png",im_out)
    # cv.imshow("im_out",im_out)


    cv.waitKey()
    cv.destroyAllWindows()

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main('PyCharm')


