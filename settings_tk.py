"""
设置界面的实现
"""

import pickle as pl
from tkinter import simpledialog as dil,\
    messagebox as mes,\
    Tk,\
    Button,\
    Label,\
    Text,\
    Entry,\
    StringVar
from tkinter.ttk import Treeview,Combobox

import camera_save as coma
import create2d

def collect_material():
    coma.collect_material()

def create_tag():
    # print(entry_id.get(),entry_sidePixels.get(),box.current())
    create2d.create_tag_img(entry_id.get(),entry_sidePixels.get(),box.current())

win = Tk()
win.title("设置")

#row 1
Label(text="标定大小",width=6,height=2).grid(row=1,column=1,columnspan=2)
Entry(width=10).grid(row=1,column=3)
#row 2
Button(text="采集标定图",command=collect_material).grid(row=2,column=1,columnspan=2)

#row 3
entry_id=StringVar()
entry_sidePixels=StringVar()
# entry_dict=StringVar()
Label(text="id:",width=3).grid(row=3,column=1)
Entry(width=5,textvariable=entry_id).grid(row=3,column=2)
Label(text="sidePixels:").grid(row=3,column=3)
Entry(width=5,textvariable=entry_sidePixels).grid(row=3,column=4)
Label(text="dict:").grid(row=3,column=5)
# Entry(width=5,textvariable=entry_dict).grid(row=3,column=6)
box= Combobox()
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
box.grid(row=3,column=6)
Button(text="生成标签",command=create_tag).grid(row=3,column=7)




win.geometry("640x480+200+200") #widthxheight+x+y



win.mainloop()