import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.pyplot import MultipleLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
    
    
def vis_angle(ann_txt,result_save_dir):
    pitch_list = []
    yaw_list = []
    roll_list  = []
    # tr_list = []
    x_list = []
    
    prefix = os.path.basename(ann_txt)
    prefix, subfix = os.path.splitext(prefix)
    # 读取头姿估计后得到的pitch、yaw、roll坐标
    with open(ann_txt, 'r')as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip().split(' ')
            # pitch, yaw, roll, tr = float(line[2]), float(line[3]), float(line[4]), float(line[5])
            pitch, yaw, roll= float(line[2]), float(line[3]), float(line[4])
            pitch_list.append(pitch)
            yaw_list.append(yaw)
            roll_list.append(roll)
            # tr_list.append(tr)
            x_list.append(i+1)

    # x_list = [float(item/25) for item in x_list]     # x轴转换为时间
    # 设置刻度数量
    # tick_x = 30
    # tick_y = 10
     
    # plt.xticks(np.arange(min(x_list), max(x_list)+1, (max(x_list)-min(x_list))/tick_x)) # 设置 x 轴刻度数量为 tick_x
    # plt.yticks(np.arange(-5, 5, 1)) # 设置 x 轴刻度数量为 tick_x

    # 绘制二维曲线图
    plt.figure(figsize=(12,9))  # 定义图的大小
    plt.xlabel("frame")     # X轴标签
    plt.ylabel("point")        # Y轴坐标标签
    plt.title(os.path.basename(ann_txt))      #  曲线图的标题
    plt.grid(True, linestyle="--", alpha=0.5)   # 设置网格
    x_major_locator=MultipleLocator(1)
    
    ax=plt.gca()
    # plt.plot(x_list, tr_list, label='tr')
    # plt.plot(x_list, roll_list, label = 'roll')
    plt.plot(x_list, yaw_list, label = 'yaw')
    plt.plot(x_list, pitch_list, label = 'pitch')
    # plt.plot(x_list, x_center_list, label = 'x_center')
    # plt.plot(x_list, y_center_list, label='y_center')
    # #在ipython的交互环境中需要这句话才能显示出来
    # plt.show()
    plt.legend()    # 设置每条曲线的图标
    plt.savefig(os.path.join(result_save_dir, prefix+ '.png'))




if __name__  == '__main__':
    
    ann_dir = 'runs/pred/exp10'
    result_save_dir = 'runs/pred/exp10/'
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
    anns = os.listdir(ann_dir)
    anns = [item for item in anns if item.endswith('.txt')]
    for ann in anns:
        ann_txt = os.path.join(ann_dir, ann)
        vis_angle(ann_txt, result_save_dir)