# 生成棋盘原图及标定的图片，存储标定数据到yaml配置文件

import numpy as np
import cv2
import os
import yaml

def generate_chessboard(cube_cm=2., pattern_size=(8, 6), scale=37.79527559055118):
    """
    generate chessboard image with given cube length, which adapts to A4 paper print
    :param cube_cm: float, single cube length in cm
    :param pattern_size: (x, y), the number of points in x, y axes in the chessboard
    :param scale: float, scale pixel/cm in A4 paper
    """
    # convert cm to pixel
    cube_pixel = cube_cm * scale
    width = round(pattern_size[0] * cube_cm * scale)
    height = round(pattern_size[1] * cube_cm * scale)

    # generate canvas
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

    # add border around the chess
    chessboard = cv2.copyMakeBorder(image, 30, 30, 30, 30, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # visualize
    win_name = "chessboard"
    cv2.imshow(win_name, chessboard)
    cv2.imwrite("pic/"+win_name+".jpg",chessboard)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calib_camera(calib_dir, pattern_size=(8, 6), draw_points=False):
    """
    calibrate camera
    :param calib_dir: str
    :param pattern_size: (x, y), the number of points in x, y axes in the chessboard
    :param draw_points: bool, whether to draw the chessboard points
    """
    # print(calib_dir,pattern_size,draw_points) #config/ (8, 6) True
    # store 3d object points and 2d image points from all the images
    object_points = []
    image_points = []

    # 3d object point coordinate
    xl = np.linspace(0, pattern_size[0], pattern_size[0], endpoint=False)
    yl = np.linspace(0, pattern_size[1], pattern_size[1], endpoint=False)
    xv, yv = np.meshgrid(xl, yl)
    object_point = np.insert(np.stack([xv, yv], axis=-1), 2, 0, axis=-1).astype(np.float32).reshape([-1, 3])

    # set termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # load image
    img_dir = calib_dir
    assert os.path.isdir(img_dir), 'Path {} is not a dir'.format(img_dir)
    imagenames = os.listdir(img_dir)
    calib_img_count=100
    for imagename in imagenames:
        if not os.path.splitext(imagename)[-1] in ['.jpg', '.png', '.bmp', '.tiff', '.jpeg']:
            continue
        img_path = os.path.join(img_dir, imagename)
        # print("img_path",img_path)
        img = cv2.imread(img_path)
        # cv2.imshow(img_path,img) # 标题不能重复，重复标题只显示一个窗口

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 找到棋盘点
        ret, corners = cv2.findChessboardCorners(img_gray, patternSize=pattern_size)
        # print(ret) # True
        # print(corners) #三维数组
        if ret:
            # add the corresponding 3d points to the summary list
            object_points.append(object_point)
            # if chessboard points are found, refine them to SubPix level (pixel location in float)
            corners_refined = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
            # add the 2d chessboard points to the summary list
            image_points.append(corners.reshape([-1, 2]))
            # visualize the points
            if draw_points:
                cv2.drawChessboardCorners(img, pattern_size, corners_refined, ret)
                if img.shape[0] * img.shape[1] > 1e6:
                    scale = round((1. / (img.shape[0] * img.shape[1] // 1e6)) ** 0.5, 3)
                    img_draw = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                else:
                    img_draw = img
                calib_img_count+=1
                filename = "calib_img/" + str(calib_img_count) + ".jpg"
                cv2.imwrite(filename, img_draw)
                cv2.imshow(filename, img_draw)

    assert len(image_points) > 0, 'Cannot find any chessboard points, maybe incorrect pattern_size has been set'
    # calibrate the camera, note that ret is the rmse of reprojection error, ret=1 means 1 pixel error
    reproj_err, k_cam, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points,
                                                                       image_points,
                                                                       img_gray.shape[::-1],
                                                                       None,
                                                                       None,
                                                                       criteria=criteria)
    # print("reproj_err",reproj_err) #0.2616373536918508
    print("k_cam",k_cam) #[[616.52996432   0.         349.19025283]
                         # [  0.         607.8280132  242.16897061]
                         # [  0.           0.           1.        ]]
    print("dist_coeffs",dist_coeffs) #[[-5.70066409e-01  6.47653659e+00  1.03422776e-02  1.53270067e-02 -2.95469369e+01]]
    # print("rvecs",rvecs)#里面有6个array([[-0.63525905],[ 0.08726233],[ 0.05195479]])这样值的数组
    # print("tvecs",tvecs)#里面有6个array([[-2.5654594 ],[ 0.89226089],[29.50347634]])这样值的数组
    save_config("config/camera_dell.yaml","camera_matrix","3","3","d",k_cam)
    # save_config("config/camera_dell.yaml", "dist_coeff", "3", "3", "d", dist_coeffs)
    # return k_cam, dist_coeffs

def save_config(path,key,rows,cols,dt,data):
    ject={key:{"rows":rows,"cols":cols,"dt":dt,"data":data}}
    file=open(path,'w+')
    file.write(yaml.dump(ject))
    file.close()

if __name__ == '__main__':
    # generate_chessboard() # 棋盘生成
    calib_camera('config/',(8,6),True) # 棋盘标定
    cv2.waitKey()
    cv2.destroyAllWindows()
