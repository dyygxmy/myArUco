"""
设置界面的实现
"""

import pickle as pl
from tkinter import simpledialog as dil, \
    messagebox as mes, \
    Tk, \
    Button, \
    Label, \
    Text, \
    Entry, \
    StringVar
from tkinter.ttk import Treeview, Combobox

import camera_save as coma
import create2d
import arUcoTest
import calib_camera


# 拍摄棋盘图片，准备标定用
def collect_material():
    coma.collect_material()


# 生成标签图像
def create_tag():
    # print(entry_id.get(),entry_sidePixels.get(),box.current())
    create2d.create_tag_img(entry_id.get(), entry_sidePixels.get(), box.current())


# 生成棋盘图像
def create_board(cube, height, width, scale):
    calib_camera.generate_chessboard(int(cube), (int(height), int(width)), float(scale))

#开始标定
def calib_start(cout_x,cout_y):
    calib_camera.calib_camera(pattern_size=(int(cout_y),int(cout_x)),draw_points=True)

# 标签识别
def identify_tag(distance):
    arUcoTest.identify_tag(int(distance))


win = Tk()
win.title("设置")

# row 1
Label(text="标定", width=6, height=2).grid(row=1, column=1, columnspan=2)

# row 2
Button(text="采集标定图", command=collect_material).grid(row=2, column=1, columnspan=2)

# row 3
entry_id = StringVar(value="2")
entry_sidePixels = StringVar(value="50")
# entry_dict=StringVar()
Label(text="id:", width=3).grid(row=3, column=1)
Entry(width=5, textvariable=entry_id).grid(row=3, column=2)
Label(text="sidePixels:").grid(row=3, column=3)
Entry(width=5, textvariable=entry_sidePixels).grid(row=3, column=4)
Label(text="dict:").grid(row=3, column=5)
# Entry(width=5,textvariable=entry_dict).grid(row=3,column=6)
box = Combobox()
box["value"] = ["DICT_4X4_50=0",
                "DICT_4X4_100=1",
                "DICT_4X4_250=2",
                "DICT_4X4_1000=3",
                "DICT_5X5_50=4",
                "DICT_5X5_100=5",
                "DICT_5X5_250=6",
                "DICT_5X5_1000=7",
                "DICT_6X6_50=8",
                "DICT_6X6_100=9",
                "DICT_6X6_250=10",
                "DICT_6X6_1000=11",
                "DICT_7X7_50=12",
                "DICT_7X7_100=13",
                "DICT_7X7_250=14",
                "DICT_7X7_1000=15"]
box.current(8)
box.grid(row=3, column=6,columnspan=4)
Button(text="生成标签", command=create_tag).grid(row=3, column=10)

# row 4
Label(text="格子大小:").grid(row=4, column=1, columnspan=2)
entry_cube = Entry(width=5, textvariable=StringVar(value="2"))
entry_cube.grid(row=4, column=3)
Label(text="cm").grid(row=4, column=4)
Label(text="宽:").grid(row=4, column=5)
entry_width = Entry(width=5, textvariable=StringVar(value="7"))
entry_width.grid(row=4, column=6)
Label(text="高:").grid(row=4, column=7)
entry_height = Entry(width=5, textvariable=StringVar(value="9"))
entry_height.grid(row=4, column=8)
Label(text="系数:").grid(row=4, column=9)
entry_scale = Entry(width=10, textvariable=StringVar(value="37.79527559055118"))
entry_scale.grid(row=4, column=10)
Button(text="生成棋盘",
       command=lambda: create_board(entry_cube.get(),
                                     entry_height.get(),
                                     entry_width.get(),
                                     entry_scale.get())
       ).grid(row=4, column=11)

# row 5
Label(text="横点:").grid(row=5,column=1)
entry_x=Entry(width=5, textvariable=StringVar(value="6"))
entry_x.grid(row=5, column=2)
Label(text="竖点:").grid(row=5,column=3)
entry_y=Entry(width=5, textvariable=StringVar(value="8"))
entry_y.grid(row=5, column=4)
Button(text="开始标定",command=lambda :calib_start(entry_x.get(),entry_y.get())).grid(row=5,column=5)


# row 6
Label(text="距离初值:").grid(row=6, column=1, columnspan=2)
entry_distance = Entry(width=5, textvariable=StringVar(value="30"))
entry_distance.grid(row=6, column=3)
Button(text="开始识别", command=lambda : identify_tag(entry_distance.get())).grid(row=6, column=4)

win.geometry("640x480+200+200")  # widthxheight+x+y

win.mainloop()
