import random
import time
import math
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import deque
from tools import kmeans_calib
from PIL import Image, ImageFont, ImageDraw


def confusion_matrix_knn(y_pred, y_true):
    C = confusion_matrix(y_true, y_pred, labels=['0','1','2','3','4']) 
    plt.matshow(C, cmap=plt.cm.Reds) # 根据最下面的图按自己需求更改颜色
    # plt.colorbar()
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()
    plt.savefig('tools/txt_files/confusion_matrix_knn.png')


def gaussian(dist, sigma = 12.5):
    """ 
    input: distance 
    return:weight
    """
    weight = np.exp(-dist**2/(2*sigma**2))
    return weight


class MyKNN:
    def __init__(self, n_neighbors):
        '''指定近邻个数'''
        self.n_neighbors = n_neighbors # 近邻数
        # self.x_new = [] # 预测数据集
        self.cls_id_list = [0, 1, 2, 3, 4] # 类别ID，注视区域分类ID
        self.que = deque(maxlen=3)   # 原始预测值保存
        self.que_predict = deque(maxlen=3)   # 多帧过滤后的预测结果
        self.que_yaw = deque(maxlen=3)
        self.forward_limit_right_left = 20.0
        self.right_left_limit_forward = 15.0

        
           
    def predict(self, x_new):
        '''预测模型, 更新近邻序列和得出测试数据所属类'''
        targetList = [] # 存储预测数据的预测所属类

        for i in tqdm(range(len(x_new))):
            tp = x_new[i]
            pitch, yaw = tp[0].item(), tp[1].item()
            # pitch -= self.pitch_calib
            # yaw -= self.yaw_calib
            distList = self.get_deftDist(tp) # 获取此时测试数据x_new[i]与预先序列的距离
            for j in range(len(self.x_train)):
                Lmax = max(distList)
                L = 0
                for k in range(len(self.x_train[j])):
                    L += (tp[k]-self.x_train[j][k])**2 # 遍历训练集，计算当前x_new[i]训练集与x_train[j]的距离
                L = math.sqrt(L)
                if L >= Lmax:
                    continue # 若当前所得距离大于预先序列距离的最大值，进入下一个训练数据
                else:
                    # 更新预先序列中的距离和对应训练数据的id
                    if j not in self.idList:
                        index = distList.index(Lmax) # 获取最大预先距离的索引
                        self.idList[index] = j # 换为当前训练数据的id
                        distList[index] = L # 更新最大预先距离
                
            classCount = {}
            for m in range(len(self.idList)): # 遍历预先序列中，计算其中的多数类，判断测试数据属于哪类
                indI = self.idList[m]
                targetI = self.y_train[indI]
                weight = gaussian(distList[m]) 
                 ## 权重值累加
                classCount[targetI] = classCount.get(targetI, 0) + weight*1 
                # 在k个近邻中，对k个近邻的预测结果，分别设置一个对应的权重，并相同的类别权重累加，最后去最大的权重对应的预测类别
            maxCount = 0
            #print(classCount)
            for key, value in classCount.items():
                if value > maxCount:
                    maxCount = value
                    classes = key
            targetList.append(classes)

        self.que_yaw.append(yaw)
        dgze_id = targetList[0]
    
        if len(self.que)<3 :
            self.que.append(dgze_id)
            dgze_id_filter = dgze_id
            self.que_predict.append(dgze_id_filter)
        else:
            self.que.append(dgze_id)
            if self.que.count(self.que[-1])==3: # 连续3帧预测结果一致
                # 所有的预测和过滤值一致
                if self.que[-1] == self.que_predict[-1] and self.que_predict[-1]==self.que_predict[-2]:    
                    # 和上一帧的过滤结果保存一致
                    dgze_id_filter = self.que_predict[-1]
                else:
                    dgze_id_filter = self.get_measure()
                    
            elif  self.que.count(self.que[-1])==2:
                # 若是连续的帧k和k-1
                if self.que[-1]== self.que[-2]:
                    dgze_id_filter = self.get_measure()
                else:
                    #不是连续两帧（1， 0， 1）和第一帧的过滤结果保持一致,
                    dgze_id_filter = self.que_predict[-2]   # = self.qeu_predict[-1]应该也是可以的
                    
            else: 
                #  self.que.count(self.que[-1])==1的情况
                dgze_id_filter = self.que_predict[-1]
            self.que_predict.append(dgze_id_filter)
          
                
        targetList[0] = dgze_id_filter
        return targetList
    
    def get_measure(self, ):
        if self.que_predict[-2]== 2 or self.que_predict[-2]==1:   
            if self.que[-1]==0:
                #right->forward和left->forward 一致
                if abs(self.que_yaw[-1]) <self.right_left_limit_forward and abs(self.que_yaw[-2])<self.right_left_limit_forward:
                    dgze_id_filter = self.que[-2]
                # left--> right 不正常，因此上一帧过滤前的值一致
                elif self.que_yaw[-1]>0 and self.que_yaw[-2]>0 and self.que_predict[-1]==1:
                    dgze_id_filter = self.que[-2]
                # right--> left 不正常，因此上一帧过滤前的值一致
                elif self.que_yaw[-1]<0 and self.que_yaw[-2]<0 and self.que_predict[-1]==2:
                    dgze_id_filter = self.que[-2]
                else:
                    dgze_id_filter = self.que_predict[-1]   #等于过滤后的预测值
            else:
                # 若是right and left ->中控台等，则和上一帧过滤前的值一致
                dgze_id_filter = self.que[-2]  
                
        elif self.que_predict[-2] == 0: 
            #forward—>left or forward->right
            if self.que[-1] == 1 or self.que[-1]==2:
                if abs(self.que_yaw[-1]) >self.forward_limit_right_left and abs(self.que_yaw[-2])>self.forward_limit_right_left:
                    dgze_id_filter = self.que[-2]
                else:
                    dgze_id_filter = self.que_predict[-1]   #等于过滤后的预测值,
            else:
                # 若forward->中控台等则和上一帧过滤前的值一致
                dgze_id_filter = self.que[-2]
        else:   
            # 中控台->right, 中控台->forward，中控台->仪表盘， 仪表盘->forward, 仪表盘->left等等；
            dgze_id_filter = self.que[-2]
            
        return dgze_id_filter
    
    
    def get_trainData(self, x_train, y_train):
        '''获取训练集数据'''
        self.x_train = x_train
        self.y_train = y_train
        self.idList = random.sample(range(0, len(x_train)), self.n_neighbors) # 获取空间大小为k的预先序列的索引
        ''' 获取空间大小为k的预先序列,k个随机的元         	
        组,k=n_neighbors'''
        #self.deftDic = {'id':self.idList, 'distance':self.distList, 'target':self.y_train}
        
    def get_deftDist(self, tp):
        '''计算测试数据与预先序列的距离'''
        list = [] # 存放预测数据与预先序列的距离，列表含有k个距离
        '''
        for i in range(len(self.x_new)):
            tp = self.x_new[i]
        '''
        for j in range(len(self.idList)):
            sum = 0
            index = self.idList[j]
            for k in range(len(tp)):
                sum += (tp[k]-self.x_train[index][k])**2 
            # 欧式距离
            sum = math.sqrt(sum)    # 未进行归一化处理
            list.append(sum)
        return list
    
    def score(self, y_pre, y_test):
        '''计算精确值'''
        count = 0
        scoreList = list(map(lambda x: x[0]-x[1], zip(y_pre, y_test)))
        for i in scoreList:
            if i == 0:
                count += 1
        score = count/len(scoreList)
        return score


def ann_hanzi(frame, cls_id, cls_name_dict, local):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype('SimHei.ttf', 30)
    text_label = 'ID:'+ cls_name_dict[cls_id]
    #  text_label = 'ID: %s  '%cls_id+ cls_name_dict[cls_id]
    draw.text(local, text_label, font=font, fill=(255, 255, 255), stroke_width=2)
    frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return frame_bgr

def ann_hanzi_1(frame, cls_id, cls_name_dict, local, color=(255, 255, 255)):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype('SimHei.ttf', 30)
    text_label = 'ID:'+ cls_name_dict[cls_id]
    #  text_label = 'ID: %s  '%cls_id+ cls_name_dict[cls_id]
    draw.text(local, text_label, font=font, fill=color, stroke_width=2)
    frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return frame_bgr

if __name__ == '__main__':
    #myDataset = {'data':[[2, 3, 0, 0], [3, 4, 0, 0], [4, 5, 0, 0], [5, 6, 0, 0]],
    #             'target':[2, 1, 0, 1]}
    #X_train, y_train = myDataset['data'], myDataset['target']
    #X_test = [[1, 2, 0, 0],[0, 0, 0, 0]]
    # 分神检测id
    cls_name_dict = {'0': '前方区域', '1': '左后视镜', '2': '右后视镜', '3': '仪表盘-中控台', '4': '不目视前方'}
    file_path = 'data/output'
    img_path = 'runs/pred/frame_save'
    img_save_path = 'runs/pred/frame_save_vis_4'
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    
    # ========测试====================
    pitch_calib = -1
    yaw_calib = -25
    
    files = os.listdir(file_path)
    files.sort(key=lambda x: int(x.split('.')[0]))   #按照数字排序
    imgs = os.listdir(img_path)
    imgs.sort(key=lambda x: int(x.split('.')[0]))   #按照数字排序
    angle_list = []
    pre_id_list = []
    y_pre_list = []
    for file in files:
        pass
        with open(os.path.join(file_path, file), 'r') as f:
            lines = f.read().splitlines()
            for i, line in enumerate(lines):
                if i>=1:
                    continue
                line = line.split()
                line = [float(item) for item in line]
                pitch, yaw, pre_id = line[0]-pitch_calib, line[1]- yaw_calib, int(line[-1])
                angle = [pitch, yaw]
                angle_list.append(angle)
                pre_id_list.append(pre_id)

    X_test = np.array(angle_list)
                
    X_train = np.load('data/npy_file/x_train_0130_5cls.npy')
    y_train = np.load('data/npy_file/y_train_0130_5cls.npy')

    

    # # 是否进行数据标准化处理
    # ss = StandardScaler()
    # X_train = ss.fit_transform(X_train)
    # X_test = ss.fit_transform(X_test)

    start = time.time()      
    knn = MyKNN(n_neighbors=3)

    knn.get_trainData(X_train, y_train)
    
    for i, value in enumerate(X_test):
        if i== 712:
            a = 1
        value = np.array([value])
        y_pre = knn.predict(value)
        y_pre_list.append(y_pre[0])
        print('--------------', i)
    # score = knn.score(y_pre, y_test)
    
    # confusion_matrix_knn
    y_pre_list = [str(item) for item in y_pre_list]
    # y_test = [str(value.item()) for value in y_test]
    # confusion_matrix_knn(y_pre, y_test)
    
    for i, id in enumerate(y_pre_list):
        
        image = cv2.imread(os.path.join(img_path, imgs[i]))
        pitch, yaw=  X_test[i][0].item(), X_test[i][1].item()
        gaze_id = int(id)
        x, y = 500, 300
        detl_y = 50
        # 设置文本参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255) # BGR颜色值，这里为绿色
        # thickness = 2

        # 在图像上绘制文本
        # cv2.putText(image, text, position, font, font_scale, color, thickness)
        cv2.putText(image, 'pitch: %.3f'%pitch, (x,y), font, 1, color)
        cv2.putText(image, 'yaw: %.3f'%yaw, (x,y+detl_y*1), font, 1, color)
        # cv2.putText(image, 'roll: %.3f'%r_pred_deg.item(), (x,y+detl_y*2), font, 1, color)
        # cv2.putText(image, 'ID: %s'%(cls_name_dict[str(gaze_id)]), (x,y+detl_y*3), font, 1, color)
        local = (x,y+detl_y*3)
        local_1 = (x,y+detl_y*4)
        if gaze_id==3:
            gaze_id=4
        # if not flag:
        pre_id = pre_id_list[i]
        if pre_id==3:
            pre_id=4
        
        image = ann_hanzi(image, str(gaze_id), cls_name_dict, local)
        if gaze_id== pre_id:
            color =(255, 255, 255)
        else:
            color = (255, 0, 0)
        image = ann_hanzi_1(image, str(pre_id), cls_name_dict, local_1, color)
        cv2.imwrite(os.path.join(img_save_path, imgs[i]), image)

    print('Holding time: ', time.time()-start) # 输出KNN运行时间，时间效率
    print('the kind of X_test: {}'.format(y_pre)) # 输出测试数据的预测类别
    # print('the score:{:.5}'.format(score)) # 输出模型精准度
