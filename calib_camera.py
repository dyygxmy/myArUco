"""
生成9*7的棋盘标定原图
标定，并将标定的图片保存，标定数据写入到yaml配置文件
"""

import numpy as np
import cv2
import os
from ruamel import yaml


def generate_chessboard(cube_cm=2., pattern_size=(9, 7), scale=37.79527559055118):
    """
    在A4纸打印分辨率为96ppi时，棋盘每个格子的宽度为cube_cm
    生成指定格子长度的棋盘图像，适用A4打印纸
    cube_cm: float 单个立方体长度，单位为厘米
    pattern_size: (x, y) 棋盘x，y轴上的点数
    scale: float 在A4纸上的刻度 像素/厘米
    """
    # 将厘米转换成像素
    cube_pixel = cube_cm * scale
    width = round(pattern_size[0] * cube_cm * scale)
    height = round(pattern_size[1] * cube_cm * scale)

    # 生成画布
    image = np.zeros([width, height, 3], dtype=np.uint8)
    image.fill(255)
    color = (255, 255, 255)
    fill_color = 0
    # drawing the chessboard
    for j in range(0, height + 1):
        y = round(j * cube_pixel)
        for i in range(0, width + 1):
            x0 = round(i * cube_pixel)
            y0 = y
            rect_start = (x0, y0)

            x1 = round(x0 + cube_pixel)
            y1 = round(y0 + cube_pixel)
            rect_end = (x1, y1)
            cv2.rectangle(image, rect_start, rect_end, color, 1, 0)
            image[y0:y1, x0:x1] = fill_color
            if width % 2:
                if i != width:
                    fill_color = (0 if (fill_color == 255) else 255)
            else:
                if i != width + 1:
                    fill_color = (0 if (fill_color == 255) else 255)

    # 棋盘四周加30像素的边
    chessboard = cv2.copyMakeBorder(image, 30, 30, 30, 30, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # 画好的图保存及显示
    win_name = "chessboard"
    cv2.imshow(win_name, chessboard)
    cv2.imwrite("pic/"+win_name+".jpg",chessboard)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def calib_camera(calib_dir, pattern_size=(8, 6), draw_points=False):
    """
    相机标定
    calib_dir: str 标定原图路径
    pattern_size: (x, y) 棋盘x，y轴上的点数 w=棋盘板一行上黑白块的数量-1，h=棋盘板一列上黑白块的数量-1
    draw_points: bool 是否要画棋盘上的点
    """
    # print(calib_dir,pattern_size,draw_points) #config/ (8, 6) True
    # store 3d object points and 2d image points from all the images
    object_points = []
    image_points = []

    # 三维坐标点
    xl = np.linspace(0, pattern_size[0], pattern_size[0], endpoint=False)
    yl = np.linspace(0, pattern_size[1], pattern_size[1], endpoint=False)
    xv, yv = np.meshgrid(xl, yl)
    object_point = np.insert(np.stack([xv, yv], axis=-1), 2, 0, axis=-1).astype(np.float32).reshape([-1, 3])

    # object_point = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    # object_point[:, : 2] = np.mgrid[0: pattern_size[0], 0: pattern_size[1]].T.reshape(-1, 2)

    # 设置端点标定
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 载入原图
    img_dir = calib_dir
    assert os.path.isdir(img_dir), 'Path {} is not a dir'.format(img_dir)

    # 提前清空标定后的图片，后面标定了存入新的
    imagenames = os.listdir(img_dir)
    calib_img_count=100
    names = os.listdir("calib_img")
    for name in names:
        os.remove("calib_img/" + name)

    for imagename in imagenames:
        if not os.path.splitext(imagename)[-1] in ['.jpg', '.png', '.bmp', '.tiff', '.jpeg']:
            continue # 排除masterImg目录内非图片的文件
        img_path = os.path.join(img_dir, imagename)
        # print("img_path",img_path)
        img = cv2.imread(img_path)
        # cv2.imshow(img_path,img) # 标题不能重复，重复标题只显示一个窗口

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gray",img_gray)
        # 找到棋盘点的2d坐标
        ret, corners = cv2.findChessboardCorners(img_gray, pattern_size,None)
        # print(ret) # True 说明找到了棋盘点
        if ret:
            # 将对应的三维坐标点放入object_points列表中
            object_points.append(object_point)
            # 如果找到棋盘上的点，将其对应到SubPix level
            corners_refined = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
            # 将二维棋盘点添加到image_points列表中
            image_points.append(corners.reshape([-1, 2]))
            # 可视化的点
            if draw_points:
                cv2.drawChessboardCorners(img, pattern_size, corners_refined, ret)
                if img.shape[0] * img.shape[1] > 1e6:
                    scale = round((1. / (img.shape[0] * img.shape[1] // 1e6)) ** 0.5, 3)
                    img_draw = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                else:
                    img_draw = img
                calib_img_count+=1
                filename = "calib_img/calib" + str(calib_img_count)[1:] + ".jpg"
                cv2.imwrite(filename, img_draw)
                cv2.imshow(filename, img_draw)

    assert len(image_points) > 0, 'Cannot find any chessboard points, maybe incorrect pattern_size has been set'

    # 相机标定，注意ret为重投影误差的rmse，ret=1表示1像素的误差
    # 根据角点的2d坐标与3d空间中的位置顺序计算相机内参和畸变参数
    reproj_err, k_cam, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points,
                                                                       image_points,
                                                                       img_gray.shape[::-1],
                                                                       None,
                                                                       None,
                                                                       criteria=criteria)
    print("reproj_err",reproj_err) #0.2616373536918508
    # print("k_cam",k_cam) #[[616.52996432   0.         349.19025283]
                         # [  0.         607.8280132  242.16897061]
                         # [  0.           0.           1.        ]]
    # print("dist_coeffs",dist_coeffs) #[[-5.70066409e-01  6.47653659e+00  1.03422776e-02  1.53270067e-02 -2.95469369e+01]]
    # print("rvecs",rvecs)#里面有6个array([[-0.63525905],[ 0.08726233],[ 0.05195479]])这样值的数组
    # print("tvecs",tvecs)#里面有6个array([[-2.5654594 ],[ 0.89226089],[29.50347634]])这样值的数组
    save_config("config/camera_dell.yaml","camera_matrix",k_cam)
    save_config("config/camera_dell.yaml", "dist_coeff", dist_coeffs)

    #计算重投影误差,误差值越小，说明矫正的结果越好
    # mean_error = 0
    # for i in range(len(object_points)):
    #     img_pts2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], k_cam, dist_coeffs)
    #     img_pts2.resize(22,2)
    #     image_points[i].resize(22,2)
    #     error = cv2.norm(image_points[i], img_pts2, cv2.NORM_L2)
    #     mean_error += error*error
    # print("errorValue",mean_error/len(object_points))

    # return k_cam, dist_coeffs

def save_config(path,key,data):
    # print("olddata",data)
    # text=[]
    # text=np.concatenate(data).tolist() # 拆成一维数组再转换成列表
    # print("newdata",text)
    object={key:{"rows":data.shape[0],"cols":data.shape[1],"dt":'d',"data":str(data.tolist())}}
    # ject = yaml.load(data, Loader=yaml.RoundTripLoader)
    write_type=None
    if key=="camera_matrix":
        write_type = 'w'
    elif key=="dist_coeff":
        write_type='a'
    with open(path,write_type, encoding="utf-8") as file:
        if key == "camera_matrix":
            file.write("%YAML:1.0\n---\n")
        yaml.dump(object,file, Dumper=yaml.RoundTripDumper)
        file.close()

if __name__ == '__main__':
    # generate_chessboard() # 棋盘生成
    calib_camera('masterImg/',(8,6),True) # 棋盘标定
    cv2.waitKey()
    cv2.destroyAllWindows()
