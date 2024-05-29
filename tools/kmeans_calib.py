from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import os
 

def z_score(arr, threshold=3):
    mean = np.mean(arr)
    std_dev = np.std(arr)
    print('std_dev:', std_dev)
    z_scores = [(x - mean) / std_dev for x in arr]
    print('z_scores: ', z_scores)
    outliers = [i for i, z in enumerate(z_scores) if abs(z) > threshold]
    return outliers

 
def _kmeans(x, file):
    # global file
    prefix, suffix = os.path.splitext(file)
    data = x
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    for i in range(n_clusters):
        plt.scatter(data[kmeans.labels_ == i, 0], data[kmeans.labels_ == i, 1], label=f'cluster{i}')

    # 聚类中心
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='red', label='cluster center')
    print(kmeans.cluster_centers_)
    print(np.bincount(kmeans.labels_))
    
    id_count_list = list(np.bincount(kmeans.labels_))
    y_center_list = list(kmeans.cluster_centers_[:, 0])
    y_center_list_c = y_center_list.copy()
    sorted_y_list = sorted(y_center_list_c, reverse=True)
    max_count_index = id_count_list.index(max(id_count_list))
    
    y_calib_value = sorted_y_list[len(sorted_y_list)//2]
    value_index = y_center_list.index(y_calib_value)
    # 校正值，不一定在类别列表的中心
    
    front_value, end_value = sorted_y_list[len(sorted_y_list)//2 -1], sorted_y_list[len(sorted_y_list)//2 +1]
    front_index, end_index = y_center_list.index(front_value), y_center_list.index(end_value)
    pitch_list = [kmeans.cluster_centers_[:, 1][end_index], kmeans.cluster_centers_[:, 1][value_index], kmeans.cluster_centers_[:, 1][front_index]]
    
    print(pitch_list)
    # flag_list = z_score(pitch_list)
    
    flag = True
    # 比较法
    if len(pitch_list)==3:
        diff0 = np.abs(pitch_list[0]- pitch_list[1])
        diff1 = np.abs(pitch_list[0]- pitch_list[2])
        diff3 = np.abs(pitch_list[1]- pitch_list[2])
        thresh = 10
        if np.abs(diff3- diff0)>thresh or np.abs(diff3- diff1)> thresh:
            flag = False
    
    pitch_calib, yaw_calib = None, None 
    if value_index== max_count_index and flag:
        # 进一步判断前方区域相邻的左，右pitch值，这三者聚类中心的pitch应该是处于同一分布的，无异常值
        # 此时的对应类别的聚类中心就为yaw, pitch初始标定值
        yaw_calib, pitch_calib = kmeans.cluster_centers_[max_count_index]
        print(yaw_calib.item())
        print(pitch_calib.item())
        # print(roll_calib.item())
        
        
    else:
        print('聚类数据不满足正常的驾驶规律，需再次采集数据标定')   
    print('=========', file) 
    plt.legend()
    plt.title(file)
    # plt.show()
    plt.savefig('runs/onnx_infer/'+ prefix+ '.png')
    plt.savefig('runs/pred/kmeans.png')
    plt.close()
    return pitch_calib, yaw_calib
  

if __name__ == '__main__':
    file_path = r'/home/aizhiqing/data1/aizhiqing/code_python/gaze_estimation/6DRepNet-1/runs/onnx_infer/exp9'
    # file_path = r'E:\项目数据\头姿估计\temp_0129\exp8'
    
    files = os.listdir(file_path)
    files = [item  for item in files if item.endswith('.txt')]
    
    for file in files:
        pt_list = []
        with open(os.path.join(file_path, file), 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                line = line.split()
                line = [float(item) for item in line]
                pt = [line[3], line[2]]
                pt_list.append(pt)

        x = np.array(pt_list)
        # 输入为[yaw, pitch] 二维数组
        pitch_calib, yaw_calib = _kmeans(x, file)