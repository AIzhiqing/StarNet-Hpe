import sys
sys.path.append('/home/aizhiqing/data1/aizhiqing/code_python/gaze_estimation/6DRepNet-1/')

import os.path
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from RepVGG_B1g2_raw_1706570467_bs256 import utils_st

 

def parse_row(row):
    # return {"fid": int(row[0]), "pitch": float(row[1]), "yaw": float(row[2]), "roll": float(row[3])}
    # return {"pitch": min_pitch_roll(float(row[1])), "yaw": float(row[2]), "roll": min_pitch_roll(float(row[3]))}
    # return [int(row[0]), float(row[1]), float(row[2]), float(row[3])]
    return [
        # int(row[0]),
        # float(row[1]),
        float(row[2]), float(row[3]),
        float(row[4]),
        # float(row[5]), float(row[6]),
        # float(row[7]), float(row[8]), float(row[9]),
        # float(row[10]), float(row[11]), float(row[12]),
    ]


def read_txt(path: str):
    with open(path, "r") as fp:
        lines = fp.readlines()

    lines = [_.strip().split() for _ in lines if _.strip()]
    # import pdb;pdb.set_trace()
    lines = [parse_row(_) for _ in lines]
    return lines


def compute_angle(front_avg_R, new_row_R):
        eps = 1e-5
        front_avg_R = front_avg_R.astype('float32')
        m1 = torch.from_numpy(np.array([front_avg_R]))
        m2 = torch.from_numpy(np.array([new_row_R]).astype('float32'))
        m_R = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3
        cos = (m_R[:, 0, 0] + m_R[:, 1, 1] + m_R[:, 2, 2] - 1) / 2
        theta = torch.acos(torch.clamp(cos, -1 + eps, 1 - eps)) * 180 / np.pi   # theta要和cos符号一致
        print('cos: %f theta: %f'%(cos, theta))
        
        return theta


if __name__ == "__main__":
    
    file_dir = 'runs/pred/exp12'
    vis_save_dir = 'runs/pred/exp12/vis_dir'
    if not os.path.exists(vis_save_dir):
        os.makedirs(vis_save_dir)
        
    files = os.listdir(file_dir)
    files = [item for i, item in enumerate(files) if item.endswith('.txt')]
    # files = [item for item in files if item.split('_')[0]=='105']
    # files = [item for item in files if item.split('_')[0]=='130'] # @ change
    # files = [item for item in files if item.split('.')[0].split('_')[-1]=='01']
    
    for  file in files:
        txt_name = os.path.join(file_dir, file)
        # txt_name = "runs/onnx_infer/exp9/130_left2.txt"
        prefix, suffix = os.path.splitext(file)
        rows = read_txt(txt_name)
        limit_value = 1500
        rows = [item for i, item in enumerate(rows) if i< limit_value]

        # 均值标定
        # rows_mean = np.array(rows)
        # pitch_mean = np.mean(rows_mean[:, 0])
        # yaw_mean = np.mean(rows_mean[:, 1])
        # row_mean = np.mean(rows_mean[:, 2])
        # print('%f %f %f'%(pitch_mean, yaw_mean, row_mean))
        
        # exit()

        # 105 01 --> -8.623263 -35.559378 2.738411
        # 105 02 --> -14.942120 -21.135660 -0.083948
        # 105 mean --> -11.7826915 -28.347519 2.654463
        front_105_mean = [-11.7826915, -28.347519, 2.654463]
        front_105_mean = [item/ 180* np.pi for item in front_105_mean]

        # 130 01 --> -15.140215 -32.909148 7.872743
        # 130 02 --> -19.835339 -22.196884 1.653770
        front_130_mean= [-17.487777 , -27.553016 , 4.7632565]
        front_130_mean = [item/ 180* np.pi for item in front_130_mean]
        
        # 测试集标定值
        calib_dict ={
                    '01': [{'pitch_calib': -2.8816}, {'yaw_calib': -24.1159}], '02': [{'pitch_calib':  -0.0301}, {'yaw_calib': -25.6468}], 
                    '20240122': [{'pitch_calib':  -10}, {'yaw_calib': -25.6468}],  '20240131': [{'pitch_calib':  -19}, {'yaw_calib': -25.6468}]
                    }

        front_01_calib = [-2.8816, -24.1159, 3.585]
        front_02_calib = [-0.0301, -25.6468, 4.741]
        front_01_calib  = [item/ 180*np.pi for item in front_01_calib]
        front_02_calib = [item/ 180* np.pi for item in front_02_calib]
        
        # 20240131
        front_0131_01_calib = [-19, -25.6468, 2]
        front_0131_02_calib = [-24, -25.6468, 5]
        front_0131_03_calib = [-26, -25.6468, -2]
        
        front_0131_01_calib  = [item/ 180*np.pi for item in front_0131_01_calib]
        front_0131_02_calib  = [item/ 180*np.pi for item in front_0131_02_calib]
        front_0131_03_calib  = [item/ 180*np.pi for item in front_0131_03_calib]
        
        # print(rows)
        df = pd.DataFrame(rows)
        # df[250:].mean()
        # 0    -9.226647
        # 1   -34.299549
        # 2     3.551485

        eps = 1e-5

        # front_avg_R = utils_st.get_R(-9.226647/180*np.pi, -34.299549/180*np.pi, 3.551485/180*np.pi)
        front_105_avg_R = utils_st.get_R(front_105_mean[0], front_105_mean[1], front_105_mean[2])
        front_130_avg_R = utils_st.get_R(front_130_mean[0], front_130_mean[1], front_130_mean[2])
        front_01_avg_R = utils_st.get_R(front_01_calib[0], front_01_calib[1], front_01_calib[2])
        front_02_avg_R = utils_st.get_R(front_02_calib[0], front_02_calib[1], front_02_calib[2])
        # 20240131
        front_0131_01_avg_R = utils_st.get_R(front_0131_01_calib[0], front_0131_01_calib[1], front_0131_01_calib[2])
        front_0131_02_avg_R = utils_st.get_R(front_0131_02_calib[0], front_0131_02_calib[1], front_0131_02_calib[2])
        front_0131_03_avg_R = utils_st.get_R(front_0131_03_calib[0], front_0131_03_calib[1], front_0131_03_calib[2])
        

        front_avg_R = front_01_avg_R   # @change
        print("front_avg_R\n", front_avg_R)

        new_rows = []
        for row in df.iterrows():
            
            # new_row_R = utils_st.get_R(row[1][0]/180*np.pi, row[1][1]/180*np.pi, row[1][2]/180*np.pi)
            # 測試：
            new_row_R = utils_st.get_R(-171.88696/180* np.pi,  -129.2953/ 180* np.pi, 155.5614/ 180* np.pi)
            theta = compute_angle(front_avg_R, new_row_R)
            print(theta)
            new_rows.append(theta.numpy()[0])

        # m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3
        # cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2

        df["rote"] = new_rows

        print(df)

        # import pdb;pdb.set_trace()
        sns.lineplot(data=df, )
        # sns.lineplot(data=df, x='fid', y='pitch', hue='species',)
        # sns.lineplot(data=df, x='fid', y='yaw')
        # sns.lineplot(data=df, x='fid', y='roll')
        # plt.savefig(os.path.join(ana_save_root, "110_left1.mp4.hpe.jpg"))
        plt.title(file)
        plt.savefig(os.path.join(vis_save_dir, prefix+ '.png'))
        # plt.show()
        print(rows[0])
        plt.close()


