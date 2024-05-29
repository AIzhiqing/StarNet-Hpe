#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import copy
import os
import cv2
import numpy as np
import torch
import onnx
from onnx import helper
import onnxruntime as ort
import random
from math import cos, sin, pi
import utils


if __name__ == '__main__':
    root = "./AFLW2000_head_crop_imgs"
    names = os.listdir(root)
    onnx_path = "./SixDRepNet_1704939428_bs80/_epoch_100.onnx"
    ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print(ort_session.get_inputs()[0].name)
    VIDEO_SRC = "/home/hy/231/surround_view/3、data/20240109AI数据/130/130-loop-li.mp4"
    cap = cv2.VideoCapture(VIDEO_SRC)
    NUMS = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # width = int(cap.get(cv2.CAP_PROP_FPS))
    print("NUMS", NUMS)
    ret, frame = cap.read()
    # import pdb;pdb.set_trace()
    out = cv2.VideoWriter("./130-loop-li.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
    # for name in names:
    for i in range(int(NUMS)):
        print(i)
        # file = os.path.join(root, name)
        # img = cv2.imread(file)
        ret, img = cap.read()
        if not ret or img is None:
            break

        cv2_img = copy.deepcopy(img)
        img = img[0:800, 600:1600, :]
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        img = np.array(img)
        img = torch.tensor(img.transpose(2, 0, 1).astype('float32')).unsqueeze(0)
        mean = [0., 0., 0.]
        std = [1.0, 1.0, 1.0]
        img = img / 255

        mean = torch.tensor(mean).reshape(1, 3, 1, 1)
        std = torch.tensor(std).reshape(1, 3, 1, 1)
        img = (img - mean) / std

        inputs = {ort_session.get_inputs()[0].name: img.numpy()}
        outs = ort_session.run(None, inputs)
        # out0, out1, out2 = outs
        # import pdb;pdb.set_trace()
        R_pred = utils.compute_rotation_matrix_from_ortho6d(torch.Tensor(outs[0]).to("cuda:0"))
        euler = utils.compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi
        p_pred_deg = euler[:, 0].cpu()
        y_pred_deg = euler[:, 1].cpu()
        r_pred_deg = euler[:, 2].cpu()

        utils.draw_axis(cv2_img, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], tdx=150, tdy=150, size=100)
        utils.plot_pose_cube(cv2_img, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], size=100)
        txt = "pitch: %.4f, yaw: %.4f, roll: %.4f" % (float(p_pred_deg[0]), float(y_pred_deg[0]), float(r_pred_deg[0]))
        cv2_img = cv2.putText(cv2_img, txt, (int(40), int(40)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
                              color=(0, 255, 0), thickness=2)

        # cv2.imshow("Test", cv2_img)
        # cv2.waitKey(0)
        # cv2.imwrite(os.path.join('output/img/', name + '.png'), cv2_img)
        out.write(cv2_img)
    out.release()


