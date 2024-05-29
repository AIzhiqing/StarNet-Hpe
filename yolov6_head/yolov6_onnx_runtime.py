import cv2
import time
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple
import os,time
from .nms import non_max_suppression
import torch


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    '''Rescale coords (xyxy) from img1_shape to img0_shape.'''
    if ratio_pad is None:  # calculate from img0_shape
        gain = [min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])]  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    # if self.scale_exact:
    #     coords[:, [0, 2]] /= gain[1]  # x gain
    # else:
    #     coords[:, [0, 2]] /= gain[0]  # raw x gain
    coords[:, [0, 2]] /= gain[1]  # x gain
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, [1, 3]] /= gain[0]  # y gain
    coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])  # x1, x2
    coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])  # y1, y2
    return coords


def yolov6_onnx_infer(session, img):
    t1= time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    conf_thres = 0.5 # 置信度0.3原始 
    iou_thres = 0.3
    names = ['head']
    colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}
    outname = [i.name for i in session.get_outputs()]
    h0, w0 = img.shape[:2]

    image = img.copy()
    image = cv2.resize(image, (224,128))
    image = image.transpose((2, 0, 1)) # HWC -> CHW
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    data = image.astype(np.float32)
    data = np.ascontiguousarray(data/255)
    out = session.run(outname,{'images':data})
    #! output[0]--> (1, 588, 4); output[1]--> (1, 588, 1)
    out = torch.Tensor(np.concatenate([out[0], np.ones((1,588,1)), out[1]], axis=2))
    # sdf
    det = non_max_suppression(out, conf_thres, iou_thres)[0]
    bboxes, scores, names = det[:, 0:4],det[:, 4], det[:, 5] 
    dw, dh = w0/224, h0/ 128
    bboxes[:, 0]*= dw
    bboxes[:, 1]*= dh
    bboxes[:, 2]*= dw
    bboxes[:, 3]*= dh
    # print('***************time: ',time.time()-t1 )
    return bboxes, scores, names