"""
根据标定的相机参数来识别标签
在标签上画3D角，边框
显示摄像头视角479*480的窗口
窗口上显示中心点的坐标，id，距离，偏角
"""
import numpy as np
import time
import cv2
import cv2.aruco as aruco
from ruamel import yaml
import json
import math
from GIGETest import getImageData_init,newsStreamSource,getSourceImages,clearCache

def identify_tag(distance_init):
    timeout=1
    base = 4500  # 50像素图案的系数 标定时像素越大tvec[0][0][2]越小，则base就会越大
    scale = 37.79527559055118
    # cube_pixel = cube_cm * scale (像素=厘米*系数)
    pixel = 50
    markerLength = pixel / scale / 100  # ArUco标记的实际物理尺寸，也就是打印出来的ArUco标记的实际尺寸，一般以米为单位
    # print(markerLength) # 0.013229166666666667
    isGIGE=True


    # 加载鱼眼镜头的yaml标定文件，检测aruco并且估算与标签之间的距离,获取偏航，俯仰，滚动
    # 加载相机纠正参数
    # cv_file = cv2.FileStorage("config/yuyan.yaml", cv2.FILE_STORAGE_READ)
    # camera_matrix = cv_file.getNode("camera_matrix").mat()
    # dist_matrix = cv_file.getNode("dist_coeff").mat()
    # print("type",type(camera_matrix),type(dist_matrix)) # type <class 'numpy.ndarray'> <class 'numpy.ndarray'>
    # cv_file.release()

    # 测试自己标定的数据
    # cv_file = cv2.FileStorage("config/camera_dell.yaml", cv2.FILE_STORAGE_READ)
    # camera_matrix = cv_file.getNode("camera_matrix").mat()
    # dist_matrix = cv_file.getNode("dist_coeff").mat()
    # 此种方式yaml里的data是一维列表，根据yaml里的行列值换算成numpy.ndarray

    # print("type1",type(camera_matrix),type(dist_matrix))
    # print("camera_matrix",camera_matrix)
    # print("dist_matrix",dist_matrix)
    # type1 <class 'numpy.ndarray'> <class 'numpy.ndarray'>
    # camera_matrix [[616.52996432   0.         349.19025283]
    #  [  0.         607.8280132  242.16897061]
    #  [  0.           0.           1.        ]]
    # dist_matrix [[-5.70066409e-01  6.47653659e+00  1.03422776e-02  1.53270067e-02
    #   -2.95469369e+01]]
    # cv_file.release()

    with open("config/camera_dell.yaml", "r") as file:
        parameter = yaml.load(file.read(), Loader=yaml.Loader)
        mtx = parameter['camera_matrix']["data"]
        dist = parameter['dist_coeff']["data"]
        # print("mtx",mtx)
        # print("dist",dist)
        camera_matrix = np.array(json.loads(mtx))
        dist_matrix = np.array(json.loads(dist))
    #     print("type2",type(camera_matrix),type(dist_matrix))
    #     print("camera_matrix",camera_matrix)
    #     print("dist_matrix",dist_matrix)
    # type2 <class 'numpy.ndarray'> <class 'numpy.ndarray'>
    # camera_matrix {'rows': 3, 'cols': 3, 'dt': 'd', 'data': [616.5299643234498, 0.0, 349.19025283346315, 0.0, 607.8280131983712, 242.16897060648043, 0.0, 0.0, 1.0]}
    # dist_matrix {'rows': 1, 'cols': 5, 'dt': 'd', 'data': [-0.5700664090180133, 6.47653658931818, 0.010342277580532295, 0.015327006665322114, -29.546936896025798]}


    # 默认cam参数
    # dist=np.array(([[-0.58650416 , 0.59103816, -0.00443272 , 0.00357844 ,-0.27203275]]))
    # newcameramtx=np.array([[189.076828   ,  0.    ,     361.20126638]
    #  ,[  0 ,2.01627296e+04 ,4.52759577e+02]
    #  ,[0, 0, 1]])
    # mtx=np.array([[398.12724231  , 0.      ,   304.35638757],
    #  [  0.       ,  345.38259888, 282.49861858],
    #  [  0.,           0.,           1.        ]])

    if isGIGE:
        camera = getImageData_init()
        streamSource = newsStreamSource(camera)
    else:
        cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text (below)

    # num = 0
    while True:
        if isGIGE:
            status,frame = getSourceImages(streamSource)
            frame_ini=np.copy(frame)
            if status == -1:
                # clearCache(streamSource, camera,False) # 流清理
                # camera = getImageData_init() # 重新开启相机开始拍摄
                # streamSource = newsStreamSource(camera) # getFrame失败后用新的流
                continue
        else:
            ret, frame = cap.read()  # 读取摄像头画面
            h1, w1 = frame.shape[:2]  # 取shape属性中的垂直像素和水平像素
            # print("h",h1,"w",w1) #h 480 w 640
            # print("ret",ret) #True
            # cv2.imshow("cap",frame) #未处理的原始照片 宽640 高480
            # print("frame",frame)
            # 测试用其他图片的纠正过程
            # frame = cv2.imread("pic/baf18385c114e62b1adee68411db79a.jpg")
            # frame=cv2.imread("config/1664176216.jpg")

            # 图像去畸变之前，我们还可以使用cv.getOptimalNewCameraMatrix()优化内参数和畸变系数，通过设定自由自由比例因子alpha。
            # 当alpha设为0的时候，将会返回一个剪裁过的将去畸变后不想要的像素去掉的内参数和畸变系数；
            # 当alpha设为1的时候，将会返回一个包含额外黑色像素点的内参数和畸变系数，并返回一个ROI用于将其剪裁掉。
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_matrix, (h1, w1), 0, (h1, w1))
            # getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, alpha, newImgSize=None, centerPrincipalPoint=None)
            # alpha=0，视场会放大，alpha=1，视场不变
            # newcameramtx和roi的值与原片没关系，用不同的原片这两值都是固定的，主要还是看camera_matrix和dist_matrix
            # print("newcameramtx",newcameramtx) #[[146.4493103    0.         312.71574961]
            # [  0.         147.02703857 250.20591536]
            # [  0.           0.           1.        ]]
            # print("roi",roi) #(0, 0, 479, 639)
            # 进行roi的crop会裁掉一部分像素
            # undistort()功能是对图像去畸变
            dst1 = cv2.undistort(frame, camera_matrix, dist_matrix, None, newcameramtx)
            # undistort(src, cameraMatrix, distCoeffs, dst=None, newCameraMatrix=None)
            # cv2.imwrite("pic/dst1.png",dst1) # 此时还是640*480
            x, y, w1, h1 = roi
            dst1 = dst1[y:y + h1, x:x + w1]  # 截取垂直0~639，水平0~479像素的图像
            # cv2.imwrite("pic/dst2.png", dst1) # 变成479*480
            frame = dst1
            # cv2.imshow("dst", frame)  # 此时纠正过的画面大小改为宽479 高480 ，周边像魔镜一样

            # 测试直接用照片来识别
            # frame = cv2.imread("pic/baf18385c114e62b1adee68411db79a.jpg")

        # 灰度化，检测aruco标签，所用字典为6×6——250
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()

        # 使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        # print("corners", corners)  # 整个二维码四个角的坐标
        # print("ids", ids)  # 二维码的结果
        # print("rejectedImgPoints", rejectedImgPoints)  # 所有检测出来的区域的四角坐标的数组
        '''testimg=frame.copy()
        print("corners",corners) # 整个二维码四个角的坐标
        print("ids", ids) # 二维码的结果
        print("rejectedImgPoints", rejectedImgPoints) #所有检测出来的区域的四角坐标的数组
        # cv2.imshow("gray",gray)
        for item in rejectedImgPoints: # 检测出来的7块区域，用红线圈起来并存储到testting.png文件
            print("item:",item)
            cv2.polylines(testimg, [np.int32(item)], True, [0, 0, 255], 3, cv2.LINE_AA)  # 红色画线，线粗3
        # cv2.imshow("rejectedImgPoints",testimg)
        cv2.imwrite("pic/testting.png",testimg)'''

        if ids is not None:  # ids有值表示识别到了
            for index in range(len(ids)):
                frame = np.copy(frame_ini)
                # print("corners", len(corners), corners)  # 整个二维码四个角的坐标
                # if len(corners) > 1:# 出现多个码的情况
                #     corners = [corners[0]] #在这里才去拆分就晚了，后面提取的值会报错
                # print("corners2", len(corners), corners)
                m = cv2.moments(corners[index])  # 计算轮廓的各阶矩，字典形式
                # print("m", m)

                #算边长像素值
                # p12 = math.sqrt(
                #     abs(corners[0][0][0][0] - corners[0][0][1][0]) ** 2 + abs(corners[0][0][0][1] - corners[0][0][1][1]) ** 2)
                # p23 = math.sqrt(
                #     abs(corners[0][0][1][0] - corners[0][0][2][0]) ** 2 + abs(corners[0][0][1][1] - corners[0][0][2][1]) ** 2)
                # p34 = math.sqrt(
                #     abs(corners[0][0][2][0] - corners[0][0][3][0]) ** 2 + abs(corners[0][0][2][1] - corners[0][0][3][1]) ** 2)
                # p41 = math.sqrt(
                #     abs(corners[0][0][3][0] - corners[0][0][0][0]) ** 2 + abs(corners[0][0][3][1] - corners[0][0][0][1]) ** 2)
                # print("p12", p12)
                # print("p23", p23)
                # print("p34", p34)
                # print("p41", p41)

                # print("center", [m["m10"] / m["m00"], m["m01"] / m["m00"]])  # 中心点的坐标
                cv2.putText(frame, "center_x: " + str(round(m["m10"] / m["m00"],2)), (0, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, "center_y: " + str(round(m["m01"] / m["m00"],2)), (0, 60), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # 获取aruco返回的rvec旋转矩阵、tvec位移矩阵
                rvec, tvec, Points4 = aruco.estimatePoseSingleMarkers(corners, markerLength, camera_matrix, dist_matrix)
                # print("Points",Points4)
                # ---解读---
                # estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs=None, tvecs=None, _objPoints=None)
                # corners:detectMarkers()返回的检测到标记的角点列表
                # markerLength:ArUco标记的实际物理尺寸，也就是打印出来的ArUco标记的实际尺寸，一般以米为单位
                # cameraMatrix:相机的内参矩阵
                # distCoeffs:相机的畸变参数
                # rvecs:vector<cv::Vec3d>类型的向量，其中每个元素为每个标记相对于相机的旋转向量
                # tvecs:vector<cv::Vec3d>类型的向量，其中每个元素为每个标记相对于相机的平移向量
                # _objPoints:每个标记角点的对应点数组

                # print("rvec",rvec) #旋转矩阵 [[[ 2.69317941 -0.03222661  0.09410393]]]
                # print("tvec",tvec) #位移矩阵 [[[-0.00870618  0.00512412  0.25546513]]]
                # 估计每个标记的姿态并返回值rvet和tvec ---不同
                # from camera coeficcients
                (rvec - tvec).any()  # get rid of that nasty numpy value array error

                # 在画面上 标注auruco标签的各轴
                for i in range(rvec.shape[0]):
                    aruco.drawAxis(frame, camera_matrix, dist_matrix, rvec[i, :, :], tvec[i, :, :], 0.03)
                    aruco.drawDetectedMarkers(frame, corners, ids)


                # 显示id标记
                cv2.putText(frame, "Id: " + str(ids[index][0]), (0, 90), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # 用来标定计算系数，不同的相机数据不一样
                # base=(30*scale)/tvec[0][0][2]
                # print("30cm-base",base) # 300像素在30cm处计算是27000，50像素就是4500


                # 距离估计
                distance = tvec[index][0][2] * base/scale   # 单位是厘米
                # 显示距离
                tance = str(round(distance, 2))
                cv2.putText(frame, 'distance:' + tance + str('cm'), (0, 120), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # 角度估计
                # 考虑Z轴（蓝色）的角度
                # 本来正确的计算方式如下，但是由于蜜汁相机标定的问题，实测偏航角度能最大达到104°所以现在×90/104这个系数作为最终角度
                deg = rvec[index][0][2] / math.pi * 180
                # deg=rvec[0][0][2]/math.pi*180*90/104
                # 旋转矩阵到欧拉角
                count=2
                R = np.zeros((3, 3), dtype=np.float64)
                # print("rvec",rvec)
                # print("R",R)
                cv2.Rodrigues(rvec[index], R)
                sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
                singular = sy < 1e-6
                if not singular:  # 偏航，俯仰，滚动
                    x = math.atan2(R[2, 1], R[2, 2])
                    y = math.atan2(-R[2, 0], sy)
                    z = math.atan2(R[1, 0], R[0, 0])
                else:
                    x = math.atan2(-R[1, 2], R[1, 1])
                    y = math.atan2(-R[2, 0], sy)
                    z = 0
                # print("x",x,"y",y,"z",z) #x 2.6949957712693364 y -0.07159143471315398 z -0.007666917250534781
                # 偏航，俯仰，滚动换成角度
                # rx = x * 180.0 / 3.141592653589793
                ry = y * 180.0 / 3.141592653589793
                # rz = z * 180.0 / 3.141592653589793

                cv2.putText(frame, 'deg_z:' + str(round(ry,2)) + str('deg'), (0, 150), font, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

                # print("偏航，俯仰，滚动",rx,ry,rz) # 154.41188349933714 -4.101887058350098 -0.4392820003316881

                ####真实坐标换算####（to do）
                # print('rvec:',rvec,'tvec:',tvec)
                # # new_tvec=np.array([[-0.01361995],[-0.01003278],[0.62165339]])
                # # 将相机坐标转换为真实坐标
                # r_matrix, d = cv2.Rodrigues(rvec)
                # r_matrix = -np.linalg.inv(r_matrix)  # 相机旋转矩阵
                # c_matrix = np.dot(r_matrix, tvec)  # 相机位置矩阵
                cv2.imshow("id:"+str(ids[index][0]), frame)
        else:  # 如果找不到id
            ##### DRAW "NO IDS" #####
            cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # 显示结果画面
            cv2.imshow("No Ids", frame)


        key = cv2.waitKey(timeout)
        if key == 27:  # 按esc键退出
            print('esc break...')
            if isGIGE:
                clearCache(streamSource, camera,True)
            else:
                cap.release()
            cv2.destroyAllWindows()
            break

        if key == ord(' '):  # 按空格键保存
            #        num = num + 1
            #        filename = "frames_%s.jpg" % num  # 保存一张图像
            filename = "data/" + str(time.time())[:10] + ".jpg"
            cv2.imwrite(filename, frame)

        if key == ord('f'): #刷新距离参数
            if ids is not None:
                # 用来标定计算系数，不同的相机数据不一样
                base=(distance_init*scale)/tvec[0][0][2]
                print("%dcm-base"%(distance_init),base) # 300像素在30cm处计算是27000，50像素就是4500
        # print("clear streamSource")
        # streamSource.contents.release(streamSource)
    # ---end while---


if __name__ == "__main__":
    identify_tag(30) # 初始距离

