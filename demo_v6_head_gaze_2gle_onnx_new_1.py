from face_detection import RetinaFace
from model import SixDRepNet, SixDStartNet
import math
import re
from matplotlib import pyplot as plt
import sys
import os
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.lib.function_base import _quantile_unchecked
from knn_dgze_gaussian_logic_1_2gle import MyKNN
from collections import deque
# from knn_dgze_v0 import MyKNN
from PIL import Image, ImageFont, ImageDraw

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import utils
import matplotlib
from PIL import Image
import time
import onnxruntime as ort
# from yolov6_head.yolov6_onnx_runtime import yolov6_onnx_infer
from yolov6_head.onnx_inference_detection_mmyolo import head_onnx_infer
from tools import kmeans_calib
from collections import deque
import pubilc_demo_infer.utils as utils_1
from loss import GeodesicLoss

# matplotlib.use('TkAgg')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument('--gpu',
                        dest='gpu_id', help='GPU device id to use [0]',
                        default=3, type=int)
    # parser.add_argument('--cam',
    #                     dest='cam_id', help='Camera device id to use [0]',
    #                     default=0, type=int)
    parser.add_argument('--srouce', 
                        dest='source', help='input img',
                        default='/home/aizhiqing/data1/aizhiqing/input_dir/head_pose_0520_08_left',#! 01
                        # default='/home/aizhiqing/data1/aizhiqing/input_dir/head_pose_0424/20240424DST'  #! TODO
                        # default='/home/aizhiqing/data1/aizhiqing/input_dir/head_pose_0428'  #! TODO
                        # default='/home/aizhiqing/data1/aizhiqing/input_dir/gaze_test/test_data_0201',
                        # default= '/home/aizhiqing/data1/aizhiqing/input_dir/20240520/dms',   #! 20240122_DMS_data 
                        )
    parser.add_argument('--save_dir',
                        help='out dir', default='runs/demo/exp1')
    parser.add_argument('--save_txt_dir',
                        help='out txt dir', default='runs/demo/text')
    parser.add_argument('--snapshot',
                        dest='snapshot', help='Name of model snapshot.',
                        # default='/home/aizhiqing/data1/aizhiqing/code_python/6DRepNet_debug_train1/output/snapshots/StartNet_1714355745_bs320/_epoch_96_1.8808_1.onnx',
                        # default='/home/aizhiqing/data1/aizhiqing/code_python/6DRepNet_debug/output/snapshots/StartNet_1714355745_bs320/_epoch_96_1.9670_1.onnx',
                        # default='/home/aizhiqing/data1/aizhiqing/code_python/6DRepNet_debug/output/snapshots/StartNet_1714382630_bs320/_epoch_100_1.9858_1.onnx', #! TODO
                        default='/home/aizhiqing/data1/aizhiqing/code_python/gaze_estimation/6DRepNet-1/weights/_epoch_100.onnx', 
                        # default='/home/aizhiqing/data1/aizhiqing/code_python/6DRepNet_debug/output/snapshots/HourglassSPP_2024_05_28_bs512_opalloss/_epoch_16_4.5860_1.onnx',
                        type=str)
    parser.add_argument('--head_onnx', help='head detect onnx model',
                        default='/home/aizhiqing/data1/aizhiqing/code_python/gaze_estimation/6DRepNet-1/weights/dms_face_hand.onnx' )
    parser.add_argument('--save_viz',
                        dest='save_viz', help='Save images with pose cube.',
                        default=False, type=bool)

    args = parser.parse_args()
    return args


def create_dir(save_dir):
    base_name = os.path.basename(save_dir)
    cout = int(base_name[3:])
    cout+=1
    base_name = base_name[:3]+ str(cout)
    save_dir = os.path.join(os.path.dirname(save_dir), base_name)
    return save_dir


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


#! 更改std 和mean, 和训练设置保持一致
normalize = transforms.Normalize(
        # mean=[0.485, 0.456, 0.406],
        # std=[0.229, 0.224, 0.225]
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0])
transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                     normalize])


if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = args.gpu_id
    # cam = args.cam_id
    source = args.source
    snapshot_path = args.snapshot
    head_onnx_model = args.head_onnx
    
    save_dir = args.save_dir
    # save_txt_dir = args.save_txt_dir
    # if not os.path.exists(save_txt_dir):
    #     os.makedirs(save_txt_dir)
    # 自动生成新的保存路径
    for _ in range(999):
        if os.path.exists(save_dir):
            save_dir = create_dir(save_dir)
        else:
            break
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # model = SixDRepNet(
    #                     backbone_name='RepVGG-A0-S2',
    #                 #    backbone_name='RepVGG-B1g2',
    #                    backbone_file='',
    #                    deploy=True,
    #                    pretrained=False)
    
    # ! StartNet  
    # model = SixDStartNet(                 
    #                     num_classes=6, nums=[120, 66, 66], 
    #                     pretrained=False, arch = 'startnet_s2')
    
    onnx_path =  args.snapshot
    ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
 
    print('Loading data.')    
    providers = ['CUDAExecutionProvider',] 
    head_face_hand_session = ort.InferenceSession(head_onnx_model, providers=providers)
    
    # detector = RetinaFace(gpu_id=gpu)
    
    # Load snapshot
    # saved_state_dict = torch.load(os.path.join(
    #     snapshot_path), map_location='cpu')

    # if 'model_state_dict' in saved_state_dict:
    #     model.load_state_dict(saved_state_dict['model_state_dict'], strict=False)   #! strict=False
    # else:
    #     model.load_state_dict(saved_state_dict)
    # #! 推理
    # # model.forward = model.forward_6d  #! 
    # model.cuda(gpu)   

    # # Test the Model
    # model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    # 注视区域估计
    decision_id_dict = {0: 'Normal', 1: 'Distraction'}
    cls_name_dict = {'0': '前方区域', '1': '左后视镜', '2': '右后视镜', '3': '仪表盘-中控台', '4': '不目视前方'}
    # cls_name_dict = {'0': '正前方', '1': '左后视镜', '2': '右后视镜', '3': '仪表盘', '4': '中控台'}
    # 测试集标定值 p_calib, y_calib, r_calib = -5.9789, -26.099, -3.5488
    calib_dict = {
                '01': [{'pitch_calib': -2.8816}, {'yaw_calib': -24.1159}, {'roll_calib': -3.5488 }],   #! 20240122_DMS_data
                '02': [{'pitch_calib':  -0.0301}, {'yaw_calib': -25.6468}, {'roll_calib': 2.5488 }],   #! 20240130 DST 
                '03': [{'pitch_calib':  -5.9789}, {'yaw_calib': -26.099,} , {'roll_calib': -3.5488 }],
                '04': [{'pitch_calib': -9.2452}, {'yaw_calib': -26.278,} , {'roll_calib': 7.1616 }],  #! 0328 测试视频
                '05': [{'pitch_calib': -20.529}, {'yaw_calib': 22.545,} , {'roll_calib': -8.2357 }],    #! 0422 测试视频
                '06': [{'pitch_calib': -7.1595}, {'yaw_calib': 24.394,} , {'roll_calib': -2.111 }],    #! 0424DST 测试视频
                '07': [{'pitch_calib': 5.1595}, {'yaw_calib': 25.394,} , {'roll_calib': 1.111 }],  #! 0428 测试视频
                '08': [{'pitch_calib': -20.47}, {'yaw_calib': 28.88,} , {'roll_calib': -9.57 }],  #! 0520 DST 测试视频
                  }  
    if os.path.isdir(source):
        base_name = os.path.basename(source)
        TEST_ID = base_name.split('_')[-2]
        direction = base_name.split('_')[-1]
        
        # TEST_ID = '08'  #! TODO
        if TEST_ID is not None:
            pitch_calib_t = calib_dict[TEST_ID][0]['pitch_calib']
            yaw_calib_t = calib_dict[TEST_ID][1]['yaw_calib']
            roll_calib_t = calib_dict[TEST_ID][2]['roll_calib']
    else:
        print('========输入需要为一个dir===========>>')
    
    k = 3
    X_train = np.load('data/npy_file/x_train_0130_5cls.npy')  # 加载KNN预先准备的数据集
    y_train = np.load('data/npy_file/y_train_0130_5cls.npy')
    knn = MyKNN(k)  # 设置k值为5
    knn.get_trainData(X_train, y_train) # 加载初始数据集
    
    # 存储一段时间内的pitch,yaw值，用于获取初始校正值
    flag = True
    maxlen=1500
    pitch_yaw_que = deque(maxlen=maxlen)
    pitch_yaw_que.clear()   # 清空缓存

    # 判断输入路径是单个文件还是文件夹目录
    files = []
    if os.path.isdir(source):
        file_list = os.listdir(source)
        files = [os.path.join(source, item) for item in file_list]
      
    elif os.path.isfile(source):
        files.append(source)
        
    loss = GeodesicLoss()
    suffix_list = ['.MP4', '.mp4', '.avi', '.ts']
    files = [item for item in files if os.path.splitext(item)[1] in  suffix_list]
    for i, file in enumerate(files):
        name = os.path.basename(file).split('.')[0].split('_')[-1]
        ext = '01'
        # if name != ext:
        #     print('=======> 检查视频，当前测试的视频后缀不是： %s'%ext)
        #     continue
        # if os.path.basename(file) != 'Normal_20240130101730_01.ts':
        #     continue
        cap = cv2.VideoCapture(file) 
        # 获取视频的帧宽和高  
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
        fps = cap.get(cv2.CAP_PROP_FPS)  
        # 获取视频的总帧数  
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        save_name = os.path.basename(file)
        # txt_name = 
        prefix, suffix = os.path.splitext(save_name)
        txt_name = prefix+ '.txt'
        bbox_txt_name = prefix+ '_bbox.txt'
        
        out = cv2.VideoWriter(os.path.join(save_dir, save_name), cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))  
        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        # frame_save_path = os.path.join(save_dir, prefix)
        # frame_save_path_crop =  os.path.join(save_dir, prefix+'_crop')
        # if not os.path.exists(frame_save_path_crop):
            # os.makedirs(frame_save_path_crop)
        frame_save_vis_path = os.path.join(save_dir, (prefix+ '_vis'))
        # if not os.path.exists(frame_save_path):
        #     os.makedirs(frame_save_path)
        # if not os.path.exists(frame_save_vis_path):   #!
        #     os.makedirs(frame_save_vis_path)
           
        # 分神
        dis_flag = 0
        dis_thresh = 30
        
        ang_thresh = 35
        angle_t = 30    #!
        pitch_t = 20
        
        # p_calib, y_calib, r_calib = -5.9789, -26.099, -3.5488
        R_calib = utils.get_R(pitch_calib_t/ 180*np.pi, yaw_calib_t/180* np.pi, roll_calib_t/ 180* np.pi)
        
        R_calib = torch.from_numpy(R_calib).to(gpu)
        R_calib = torch.unsqueeze(R_calib, 0)
        
        with torch.no_grad():
            frame_id = 0
            while True:
                ret, frame = cap.read()
                if frame is None:
                    break
                image = frame.copy()
                w, h = frame.shape[1], frame.shape[0]
                # faces = detector(frame)
                bboxes = head_onnx_infer(head_face_hand_session, frame) # yolov6人头检测
                frame_id+=1
                
                if len(bboxes)<=0:
                    out.write(image)
                    continue
                # 只将面积最大的人头检测框作为输入，只关心驾驶员的头部姿态角度
                x_center_list = []
                box_aera_list  = []
                for i, bbox in enumerate(bboxes):
                    x_min, y_min, x_max, y_max = bbox   # yolov6-head中的bbox
                    x_center = (x_min+ x_max) / 2
                    x_center_list.append(x_center)
                    w_box, h_box = max(0, (x_max - x_min)), max(0, (y_max-y_min))
                    box_aera = w_box* h_box
                    box_aera_list.append(box_aera)
                    
                max_box_aera = max(box_aera_list)
                max_id = box_aera_list.index(max_box_aera)
                if direction=="left":
                    max_id_c = x_center_list.index(min(x_center_list)) #! TODO 根据驾驶员位置修改
                elif direction== "right":
                    max_id_c = x_center_list.index(max(x_center_list))  #!  根据驾驶员具体位置更改
                if max_id_c == max_id:
                    bbox = bboxes[max_id_c]
                
                    x_min = int(bbox[0])
                    y_min = int(bbox[1])
                    x_max = int(bbox[2])
                    y_max = int(bbox[3])
                    bbox_width = abs(x_max - x_min)
                    bbox_height = abs(y_max - y_min)

                    x_min = max(0, x_min-int(0.2*bbox_height))
                    y_min = max(0, y_min-int(0.2*bbox_width))
                    x_max = min(x_max+int(0.2*bbox_height), w)
                    y_max = min(y_max+int(0.2*bbox_width), h)
                    
                    
                    bbox_list = [frame_id, x_min, y_min, x_max, y_max]
                    bbox_value = [str(item) for item in bbox_list]
                    bbox_value = str.join(' ', bbox_value)
                    with open(os.path.join(save_dir, bbox_txt_name), 'a')as f:
                        f.write(bbox_value+ '\n')
                    
                    # cv2.imwrite(os.path.join(frame_save_path, '%d.jpg'%frame_id), frame)
                    
                    img = frame[y_min:y_max, x_min:x_max]
                    # cv2.imwrite(os.path.join(frame_save_path, '%d.jpg'%frame_id), img)
                    
                    # cv2.imwrite('runs/pred/test.png', img)
                    img = Image.fromarray(img)
                    img = img.convert('RGB')
                    img = transformations(img)

                    img = torch.Tensor(img[None, :])
                    inputs = {ort_session.get_inputs()[0].name: img.numpy()}
                    outs = ort_session.run(None, inputs)
                    # R_pred = model(img) #! 
                    # end = time.time()
                    # print('Head pose estimation: %2f ms' % ((end - start)*1000.))
                    R_pred = utils.compute_rotation_matrix_from_ortho6d(torch.Tensor(outs[0]).to(gpu)
                                                                        )
                    theta = loss.forward(R_calib, R_pred).to('cpu').item() /np.pi* 180
                    
                    euler = utils.compute_euler_angles_from_rotation_matrices(
                        R_pred)*180/np.pi
                    p_pred_deg = euler[:, 0].cpu()
                    y_pred_deg = euler[:, 1].cpu()
                    r_pred_deg = euler[:, 2].cpu()

                    p_pred_t = p_pred_deg.item()
                    y_pred_t = y_pred_deg.item()
                    #! 根据ptich，重新设定阈值
                    if p_pred_t- pitch_calib_t >pitch_t and -20< y_pred_t- yaw_calib_t< 30:
                        thresh = angle_t
                    else:
                        thresh = ang_thresh
                    if TEST_ID is None:
                        # 初始校正角度值
                        # que存储一段时间内的yaw，和pitch值,跳帧存入
                        pt_valib = [y_pred_deg.item(), p_pred_deg.item()]
                        if (frame_id % 2==0):
                            # 跳帧存入
                            pitch_yaw_que.append(pt_valib)
                        
                        if len(pitch_yaw_que)>= maxlen and flag:
                            x_data = np.array(pitch_yaw_que)
                            pitch_calib, yaw_calib = kmeans_calib._kmeans(x_data, os.path.basename(file))
                            if pitch_calib and yaw_calib is not None:
                                flag = False
                            else:
                                # 清空deque中的值，重新存储数据标定
                                pitch_yaw_que.clear()
                        
                        # 注视区域估计
                        if not flag:
                            p_pred_t = p_pred_t - pitch_calib
                            y_pred_t = y_pred_t - yaw_calib
                            print('kmeans 初始值标定成功：-------------->%.4f %.4f'%(pitch_calib, yaw_calib))
                    else:
                        p_pred_t = p_pred_t - pitch_calib_t
                        y_pred_t = y_pred_t - yaw_calib_t                    
                
                    x_test = np.array(list([p_pred_t, y_pred_t]))
                    y_pre = knn.predict([x_test])
                    gaze_id = y_pre[0]
                    
                    # 分神判定
                    if gaze_id== 3 or gaze_id==4:
                        dis_flag+=1
                    else:
                        dis_flag-=1
                    dis_flag = max(0, dis_flag)
                    dis_flag = min(dis_flag, dis_thresh)
                    # 连续报警k帧：
                    print('dif_flag----------->: ', dis_flag)
                    
                    if dis_flag >= dis_thresh:
                        decision_id = 1
                    else:
                        decision_id = 0
                    
                    x, y = 500, 150
                    detl_y = 30
                    # 设置文本参数
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    color = (255, 255, 255) # BGR颜色值，这里为绿色
                    # thickness = 2
                    image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 1)
                    # 在图像上绘制文本
                    # cv2.putText(image, text, position, font, font_scale, color, thickness)  
                    cv2.putText(image, 'pitch: %.3f'%p_pred_deg.item(), (x,y), font, font_scale, color)
                    cv2.putText(image, 'yaw: %.3f'%y_pred_deg.item(), (x,y+detl_y*1), font, font_scale, color)
                    cv2.putText(image, 'roll: %.3f'%r_pred_deg.item(), (x,y+detl_y*2), font, font_scale, color)
                    if theta> thresh:
                        color=(0, 0, 255)
                    cv2.putText(image, 'theta: %.3f'%theta, (x,int(y+detl_y*3.5)), font, font_scale, color)
                    color_id = (255, 255, 255)
                    if decision_id==1:
                        color_id = (0, 0, 255)
                    # cv2.putText(image, 'ID: %s'%(decision_id_dict[decision_id]), (x,y+detl_y*4), font, font_scale, color_id)
                    local = (x,y+detl_y*3)
                    if gaze_id==3:
                        gaze_id=4
                    if not flag or TEST_ID is not None:
                        # image = ann_hanzi(image, str(gaze_id), cls_name_dict, local)
                        pass
                    # 将结果保存到文件
                    value_text = [frame_id, gaze_id, p_pred_deg.item(), y_pred_deg.item(), r_pred_deg.item()]
                    value_text = [str(item) for item in value_text]
                    value_text = str.join(' ', value_text)
                    with open(os.path.join(save_dir, txt_name), 'a') as f:
                        f.write(value_text+ '\n')
                    # utils.draw_axis(frame, y_pred_deg, p_pred_deg, r_pred_deg, left+int(.5*(right-left)), top, size=100)
                    # utils_1.plot_pose_cube(image,  y_pred_deg, p_pred_deg, r_pred_deg, x_min + int(.5*(
                    #     x_max-x_min)), y_min + int(.5*(y_max-y_min)), size=bbox_width)
                    
                    utils_1.draw_axis(image, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], tdx=150, tdy=150, size=100)
                    utils_1.plot_pose_cube(image, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], size=100)
                    
                # cv2.imwrite(os.path.join(frame_save_vis_path, '%d.jpg'%frame_id), image)  #!
                out.write(image)
                # cv2.imshow("Demo", frame)
                # cv2.waitKey(5)
        out.release()
        cap.release()

