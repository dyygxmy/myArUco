import numpy as np
import time
import cv2
import cv2.aruco as aruco
import math

# 加载鱼眼镜头的yaml标定文件，检测aruco并且估算与标签之间的距离,获取偏航，俯仰，滚动

# 加载相机纠正参数
cv_file = cv2.FileStorage("config/yuyan.yaml", cv2.FILE_STORAGE_READ)
camera_matrix = cv_file.getNode("camera_matrix").mat()
dist_matrix = cv_file.getNode("dist_coeff").mat()
cv_file.release()

# 默认cam参数
# dist=np.array(([[-0.58650416 , 0.59103816, -0.00443272 , 0.00357844 ,-0.27203275]]))
# newcameramtx=np.array([[189.076828   ,  0.    ,     361.20126638]
#  ,[  0 ,2.01627296e+04 ,4.52759577e+02]
#  ,[0, 0, 1]])
# mtx=np.array([[398.12724231  , 0.      ,   304.35638757],
#  [  0.       ,  345.38259888, 282.49861858],
#  [  0.,           0.,           1.        ]])


cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text (below)

# num = 0
while True:
    ret, frame = cap.read()# 读取摄像头画面
    h1, w1 = frame.shape[:2] #取shape属性中的垂直像素和水平像素
    # print("h",h1,"w",w1) #h 480 w 640
    # print("ret",ret) #True
    # cv2.imshow("cap",frame) #未处理的原始照片 宽640 高480

    # 测试用其他图片的纠正过程
    # frame = cv2.imread("pic/baf18385c114e62b1adee68411db79a.jpg")

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
    #进行roi的crop会裁掉一部分像素
    #undistort()功能是对图像去畸变
    dst1 = cv2.undistort(frame, camera_matrix, dist_matrix, None, newcameramtx)
    #undistort(src, cameraMatrix, distCoeffs, dst=None, newCameraMatrix=None)
    # cv2.imwrite("pic/dst1.png",dst1) # 此时还是640*480
    x, y, w1, h1 = roi
    dst1 = dst1[y:y + h1, x:x + w1] # 截取垂直0~639，水平0~479像素的图像
    # cv2.imwrite("pic/dst2.png", dst1) # 变成479*480
    frame = dst1
    cv2.imshow("dst",frame) #此时纠正过的画面大小改为宽479 高480 ，周边像魔镜一样

    # 测试直接用照片来识别
    # frame = cv2.imread("pic/baf18385c114e62b1adee68411db79a.jpg")

    # 灰度化，检测aruco标签，所用字典为6×6——250
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    # 使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
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

    if ids is not None:  # ids有值
        # 获取aruco返回的rvec旋转矩阵、tvec位移矩阵
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_matrix)
        # 估计每个标记的姿态并返回值rvet和tvec ---不同
        # rvec为旋转矩阵，tvec为位移矩阵
        # from camera coeficcients
        (rvec - tvec).any()  # get rid of that nasty numpy value array error
        # print(rvec)

        # 在画面上 标注auruco标签的各轴
        for i in range(rvec.shape[0]):
            aruco.drawAxis(frame, camera_matrix, dist_matrix, rvec[i, :, :], tvec[i, :, :], 0.03)
            aruco.drawDetectedMarkers(frame, corners, ids)

        ###### 显示id标记 #####
        cv2.putText(frame, "Id: " + str(ids), (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        ###### 角度估计 #####
        # print(rvec)
        # 考虑Z轴（蓝色）的角度
        # 本来正确的计算方式如下，但是由于蜜汁相机标定的问题，实测偏航角度能最大达到104°所以现在×90/104这个系数作为最终角度
        deg = rvec[0][0][2] / math.pi * 180
        # deg=rvec[0][0][2]/math.pi*180*90/104
        # 旋转矩阵到欧拉角
        R = np.zeros((3, 3), dtype=np.float64)
        cv2.Rodrigues(rvec, R)
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
        # 偏航，俯仰，滚动换成角度
        rx = x * 180.0 / 3.141592653589793
        ry = y * 180.0 / 3.141592653589793
        rz = z * 180.0 / 3.141592653589793

        cv2.putText(frame, 'deg_z:' + str(ry) + str('deg'), (0, 140), font, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        # print("偏航，俯仰，滚动",rx,ry,rz)

        ###### 距离估计 #####
        distance = ((tvec[0][0][2] + 0.02) * 0.0254) * 100  # 单位是米
        # distance = (tvec[0][0][2]) * 100  # 单位是米

        # 显示距离
        cv2.putText(frame, 'distance:' + str(round(distance, 4)) + str('m'), (0, 110), font, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        ####真实坐标换算####（to do）
        # print('rvec:',rvec,'tvec:',tvec)
        # # new_tvec=np.array([[-0.01361995],[-0.01003278],[0.62165339]])
        # # 将相机坐标转换为真实坐标
        # r_matrix, d = cv2.Rodrigues(rvec)
        # r_matrix = -np.linalg.inv(r_matrix)  # 相机旋转矩阵
        # c_matrix = np.dot(r_matrix, tvec)  # 相机位置矩阵

    else:  # 如果找不到id
        ##### DRAW "NO IDS" #####
        cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 显示结果画面
    cv2.imshow("endframe", frame)

    key = cv2.waitKey(5000)

    if key == 27:  # 按esc键退出
        print('esc break...')
        cap.release()
        cv2.destroyAllWindows()
        break

    if key == ord(' '):  # 按空格键保存
        #        num = num + 1
        #        filename = "frames_%s.jpg" % num  # 保存一张图像
        filename = "data/" + str(time.time())[:10] + ".jpg"
        cv2.imwrite(filename, frame)

    # break
