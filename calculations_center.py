"""corners [array([[[131., 186.],
        [409., 171.],
        [423., 438.],
        [140., 448.]]], dtype=float32)]
corners [array([[[148., 161.],
        [434., 155.],
        [440., 428.],
        [151., 431.]]], dtype=float32)]"""

import numpy as np
import cv2
corners1 = np.array([[[131., 186.],
        [409., 171.],
        [423., 438.],
        [140., 448.]]], dtype=np.float32)
corners2 = np.array([[[148., 161.],
        [434., 155.],
        [440., 428.],
        [151., 431.]]], dtype=np.float32)

# for item in corners1:

center1=(corners1[0][0]+corners1[0][2])*0.5
print("1adn3",center1)
center2=(corners1[0][1]+corners1[0][3])*0.5
print("2and4",center2)
center3=(center1+center2)*0.5
print("center",center3)

m = cv2.moments(corners1[0]) #计算轮廓的各阶矩，字典形式
print("m",m)
print("center2",m["m10"]/m["m00"],m["m01"]/m["m00"])
