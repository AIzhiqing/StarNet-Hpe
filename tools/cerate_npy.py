import os
import random
from tqdm import tqdm

# os.makedirs('yolo_v6')
# with open('test.txt', 'r')as f:
#     line_list = f.read().splitlines() # 方法用于读取文件对象 f 的全部内容，并以换行符为分隔符将内容分割成以每一行为元素的列表。
#     # 相当于 
#     # line_list = [line.strip() for line in f.readlines()]
    
import numpy as np

ann_list = []
angle_list = []
with open('data/knn_train_data_0125.txt', 'r')as f:
    lines = f.read().splitlines()
    random.shuffle(lines)
    for line in tqdm(lines):
        line = line.split(' ')
        cls_id = int(line[0])
        if cls_id not in [0, 1, 2, 3, 4]:
            continue
        # pitch, yaw, roll
        angle = [float(line[1]), float(line[2])]    # pitch, yaw
        # angle = [float(line[1]), float(line[2]), float(line[3])]
        ann_list.append(cls_id)
        angle_list.append(angle)
# pitch, yaw, roll
np.save('data/npy_file/x_train_0130_5cls.npy', np.array(angle_list))
np.save('data/npy_file/y_train_0130_5cls.npy', np.array(ann_list))
