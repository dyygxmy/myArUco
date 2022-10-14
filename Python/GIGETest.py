
from ImageConvert import *
from MVSDK import *
import struct
import time
import datetime

g_cameraStatusUserInfo = b"statusInfo"


class BITMAPFILEHEADER(Structure):
    _fields_ = [
        ('bfType', c_ushort),
        ('bfSize', c_uint),
        ('bfReserved1', c_ushort),
        ('bfReserved2', c_ushort),
        ('bfOffBits', c_uint),
    ]


class BITMAPINFOHEADER(Structure):
    _fields_ = [
        ('biSize', c_uint),
        ('biWidth', c_int),
        ('biHeight', c_int),
        ('biPlanes', c_ushort),
        ('biBitCount', c_ushort),
        ('biCompression', c_uint),
        ('biSizeImage', c_uint),
        ('biXPelsPerMeter', c_int),
        ('biYPelsPerMeter', c_int),
        ('biClrUsed', c_uint),
        ('biClrImportant', c_uint),
    ]


# 调色板，只有8bit及以下才需要
# color palette
class RGBQUAD(Structure):
    _fields_ = [
        ('rgbBlue', c_ubyte),
        ('rgbGreen', c_ubyte),
        ('rgbRed', c_ubyte),
        ('rgbReserved', c_ubyte),
    ]

# 获取系统单例
# get system instance
system = pointer(GENICAM_System())
nRet = GENICAM_getSystemInstance(byref(system))
if ( nRet != 0 ):
    print("getSystemInstance fail!")

# 发现相机
# discover camera
cameraList = pointer(GENICAM_Camera())
cameraCnt = c_uint()
nRet = system.contents.discovery(system, byref(cameraList), byref(cameraCnt), c_int(GENICAM_EProtocolType.typeAll));
if ( nRet != 0 ):
    print("discovery fail!")
elif cameraCnt.value < 1:
    print("discovery no camera!")
else:
    print("cameraCnt: " + str(cameraCnt.value))


# 显示相机信息
# print camera info
for index in range(0, cameraCnt):
    camera = cameraList[index]
    print("\nCamera Id = " + str(index))
    print("Key           = " + str(camera.getKey(camera)))
    print("vendor name   = " + str(camera.getVendorName(camera)))
    print("Model  name   = " + str(camera.getModelName(camera)))
    print("Serial number = " + str(camera.getSerialNumber(camera)))

camera = cameraList[0]

# 连接相机
# connect camera
nRet = camera.connect(camera, c_int(GENICAM_ECameraAccessPermission.accessPermissionControl))
if (nRet != 0):
    print("camera connect fail!")
else:
    print("camera connect success.")

# 注册相机连接状态回调
# subscribe camera connection status change
# nRet = subscribeCameraStatus(camera)
# if (nRet != 0):
#     print("subscribeCameraStatus fail!")