import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from matplotlib.pyplot import MultipleLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import deque
    
# import sys
# sys.path.append('/home/aizhiqing/data1/aizhiqing/code_python/gaze_estimation/6DRepNet-1')

def vis_angle(ann_txt,image, result_save_dir):
    prefix = os.path.basename(ann_txt)
    prefix, subfix = os.path.splitext(prefix)
    # 读取头姿估计后得到的pitch、yaw、roll坐标
    with open(ann_txt, 'r')as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i> 0:
                continue
            line = line.strip().split(' ')
            # pitch, yaw, roll, tr = float(line[2]), float(line[3]), float(line[4]), float(line[5])
            pitch, yaw, roll, theta= float(line[0]), float(line[1]), float(line[2]), float(line[3])
            
            
            # 分神判定 02
            if theta>= angle_thresh:
                dis_flag+=1
                frame_flag=0
            else:
                dis_flag-=1
                frame_flag+=1
            frame_flag =min(frame_flag, frame_limit+1)
            if frame_flag > frame_limit:
                dis_flag = 0
                
            dis_flag = max(0, dis_flag)
            dis_flag = min(dis_flag, dis_thresh+ filter_value)
            
            
            # 连续报警k帧：
            print('dif_flag----------->: ', dis_flag)
            
            if dis_flag >= dis_thresh:
                decision_id = 1
            else:
                decision_id = 0

            x, y = 460, 100
            detl_y = 30
            # 设置文本参数
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            color = (255, 255, 255) # BGR颜色值，这里为绿色
            # thickness = 2

            # 在图像上绘制文本
            # cv2.putText(image, text, position, font, font_scale, color, thickness)  
            # cv2.rectangle(image, (x_min,y_min), (x_max, y_max), (0, 255, 0), 1)
            cv2.putText(image, 'pitch: %.3f'%pitch, (x,y), font, font_scale, color)
            cv2.putText(image, 'yaw: %.3f'%yaw, (x,y+detl_y*1), font, font_scale, color)
            cv2.putText(image, 'roll: %.3f'%roll, (x,y+detl_y*2), font, font_scale, color)
            cv2.putText(image, 'tr: %.3f'%theta, (x,y+detl_y*3), font, font_scale, color)
            # cv2.putText(image, 'theta: %.3f'%theta.item(), (x,y+detl_y*4), font, font_scale, color)
            cv2.putText(image, 'flag: %d'%dis_flag, (x+230,y+detl_y*4), font, font_scale, (0, 0, 255))
            cv2.putText(image, 'limit: %d'%frame_flag, (x+230,y+detl_y*5), font, font_scale, (0, 0, 255))
            color_id = (255, 255, 255)
            if decision_id==1:
                color_id = (0, 0, 255)
            cv2.putText(image, '%s'%decision_id_dict[decision_id], (x,y+detl_y*5), font, font_scale, color_id)
            cv2.imwrite(result_save_dir, image)





if __name__  == '__main__':
    
    ann_dir = 'data/ywj/data2'
    img_dir  = '/home/aizhiqing/data1/aizhiqing/code_python/gaze_estimation/6DRepNet-1/runs/onnx_infer/exp71'
    result_save_dir = os.path.join(img_dir, 'vis_banduan_20')
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
        
    ann_files = os.listdir(ann_dir)
    for ann_file in ann_files:
        print(ann_file)
        anns = os.listdir(os.path.join(ann_dir, ann_file))
        name = ann_file.split('_')[:-1]
        name = str.join('_', name)+ '_vis'
        
        imgs  = os.listdir(os.path.join(img_dir, name))
        imgs = [item for item in imgs if item.endswith('.jpg')]
        anns = [item for item in anns if item.endswith('.txt')]
        anns.sort(key=lambda x: int(x.split('.')[0]))   #按照数字排序
        
        pitch_list = []
        yaw_list = []
        roll_list  = []
        angle_list = []
        x_list = []
        decision_id_dict = {0: 'Normal', 1: 'Distraction'}
        
        # 分神
        dis_flag = 0
        dis_thresh = 60 # angle_thresh較小，則dis_thresh 需要設置的更大一些
        filter_value = 3   # 滤波帧数
        angle_thresh = 25
        # frame_limit > filter_values
        frame_limit = 3 
        frame_flag =0
        # 统计分神次数
        DIS_COUNT = 0
        DIS_DECISION = 0
        maxlen=6
        decision_que = deque(maxlen=maxlen)   # 原始预测值保存
        
        for frame_id, ann in enumerate(anns):
            prefix, suffix = os.path.splitext(ann)
            ann_txt = os.path.join(os.path.join(ann_dir, ann_file), ann)
            img_file = prefix
            if img_file not in imgs:
                continue

            image = cv2.imread(os.path.join(os.path.join(img_dir,name), img_file))
            save_path = os.path.join(result_save_dir, name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_file = os.path.join(save_path,  img_file)
            # vis_angle( ann_txt, image,  save_file)
            
            prefix = os.path.basename(ann_txt)
            prefix, subfix = os.path.splitext(prefix)
            # 读取头姿估计后得到的pitch、yaw、roll坐标
            with open(ann_txt, 'r')as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if i> 0:
                        continue
                    line = line.strip().split(' ')
                    # pitch, yaw, roll, tr = float(line[2]), float(line[3]), float(line[4]), float(line[5])
                    pitch, yaw, roll, theta= float(line[0]), float(line[1]), float(line[2]), float(line[3])
                    
                    
                    # 分神判定 02
                    if theta>= angle_thresh:
                        dis_flag+=1
                        frame_flag=0
                    else:
                        dis_flag-=1
                        frame_flag+=1
                    frame_flag =min(frame_flag, frame_limit+1)
                    if frame_flag > frame_limit:
                        dis_flag = 0
                        
                    dis_flag = max(0, dis_flag)
                    dis_flag = min(dis_flag, dis_thresh+ filter_value)
                    
                    
                    # 连续报警k帧：
                    # print('dif_flag----------->: ', dis_flag)
                    
                    if dis_flag >= dis_thresh:
                        decision_id = 1
                        DIS_DECISION+=1
                    else:
                        decision_id = 0

                    decision_que.append(decision_id)
                    if len(decision_que)==maxlen:
                        pass
                        if decision_que[-1]==1 and decision_que.count(0)== maxlen-1:
                            DIS_COUNT=+1
                            print('-'*10, frame_id+1)
                        

                    x, y = 260, 100
                    detl_y = 30
                    # 设置文本参数
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    color = (255, 255, 255) # BGR颜色值，这里为绿色
                    # thickness = 2

                    # 在图像上绘制文本
                    # cv2.putText(image, text, position, font, font_scale, color, thickness)  
                    # cv2.rectangle(image, (x_min,y_min), (x_max, y_max), (0, 255, 0), 1)
                    cv2.putText(image, 'pitch: %.3f'%pitch, (x,y), font, font_scale, color)
                    cv2.putText(image, 'yaw: %.3f'%yaw, (x,y+detl_y*1), font, font_scale, color)
                    cv2.putText(image, 'roll: %.3f'%roll, (x,y+detl_y*2), font, font_scale, color)
                    cv2.putText(image, 'theta: %.3f'%theta, (x,y+detl_y*4), font, font_scale, color)
                    # cv2.putText(image, 'theta: %.3f'%theta.item(), (x,y+detl_y*4), font, font_scale, color)
                    cv2.putText(image, 'flag: %d'%dis_flag, (x,y+detl_y*6), font, font_scale, (0, 0, 255))
                    cv2.putText(image, 'limit: %d'%frame_flag, (x, y+detl_y*7), font, font_scale, (0, 0, 255))
                    color_id = (255, 255, 255)
                    if decision_id==1:
                        color_id = (0, 0, 255)
                    cv2.putText(image, '%s'%decision_id_dict[decision_id], (x,y+detl_y*5), font, font_scale, color_id)
                    # cv2.imwrite(save_file, image)  