"""
coding:utf-8
@Function: 
@Time:2023/7/14 10:31
@Author: wanghonggang
"""
import onnxruntime as ort
import cv2
import numpy as np
import torch
from typing import List, Tuple, Union
from torch import Tensor
from torch.nn.modules.utils import _pair
from mmengine.structures import InstanceData
import time
import torchvision


def xywh2xyxy(x):
    '''Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right.'''
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
    Args:
        prediction: (tensor), with shape [N, 5 + num_classes], N is the number of bboxes.
        conf_thres: (float) confidence threshold.
        iou_thres: (float) iou threshold.
        classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
        agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
        multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
        max_det:(int), max number of output bboxes.

    Returns:
         list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
    """
    # import pdb;pdb.set_trace()
    num_classes = prediction.shape[2] - 5  # number of classes
    pred_candidates = torch.logical_and(prediction[..., 4] > conf_thres, torch.max(prediction[..., 5:], axis=-1)[0] > conf_thres)  # candidates
    # Check the parameters.
    assert 0 <= conf_thres <= 1, f'conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided.'
    assert 0 <= iou_thres <= 1, f'iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided.'

    # Function settings.
    max_wh = 4096  # maximum box width and height
    max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
    time_limit = 10.0  # quit the function when nms cost time exceed the limit time.
    multi_label &= num_classes > 1  # multiple labels per box

    tik = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for img_idx, x in enumerate(prediction):  # image index, image inference
        x = x[pred_candidates[img_idx]]  # confidence
        # If no box remains, skip the next process.
        if not x.shape[0]:
            continue
        # confidence multiply the objectness
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # (center x, center y, width, height) to (x1, y1, x2, y2)
        # box = xywh2xyxy(x[:, :4])
        box = x[:, :4]

        # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
        if multi_label:
            box_idx, class_idx = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[box_idx], x[box_idx, class_idx + 5, None], class_idx[:, None].float()), 1)
        else:  # Only keep the class with highest scores.
            conf, class_idx = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, class_idx.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class, only keep boxes whose category is in classes.
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        num_box = x.shape[0]  # number of boxes
        if not num_box:  # no boxes kept.
            continue
        elif num_box > max_nms:  # excess max boxes' number.

            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + class_offset, x[:, 4]  # boxes (offset by class), scores
        keep_box_idx = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        #print(iou_thres,"iou")

        if keep_box_idx.shape[0] > max_det:  # limit detections
            keep_box_idx = keep_box_idx[:max_det]

        output[img_idx] = x[keep_box_idx]
        #print(output)
        if (time.time() - tik) > time_limit:
            print(f'WARNING: NMS cost time exceed the limited {time_limit}s.')
            break  # time limit exceeded

    return output


def _meshgrid(x, y, row_major: bool = True):
    yy, xx = torch.meshgrid(y, x)
    if row_major:
        # warning .flatten() would cause error in ONNX exporting
        # have to use reshape here
        return xx.reshape(-1), yy.reshape(-1)
    else:
        return yy.reshape(-1), xx.reshape(-1)

def grid_priors(featmap_sizes: List[Tuple],
                dtype: torch.dtype = torch.float32,
                device='cuda',
                with_stride: bool = False):
    """Generate grid points of multiple feature levels.

    Args:
        featmap_sizes (list[tuple]): List of feature map sizes in
            multiple feature levels, each size arrange as
            as (h, w).
        dtype (:obj:`dtype`): Dtype of priors. Defaults to torch.float32.
        device (str | torch.device): The device where the anchors will be
            put on.
        with_stride (bool): Whether to concatenate the stride to
            the last dimension of points.

    Return:
        list[torch.Tensor]: Points of  multiple feature levels.
        The sizes of each tensor should be (N, 2) when with stride is
        ``False``, where N = width * height, width and height
        are the sizes of the corresponding feature level,
        and the last dimension 2 represent (coord_x, coord_y),
        otherwise the shape should be (N, 4),
        and the last dimension 4 represent
        (coord_x, coord_y, stride_w, stride_h).
    """
    num_levels = len(featmap_sizes)
    multi_level_priors = []
    for i in range(num_levels):
        priors = single_level_grid_priors(
            featmap_sizes[i],
            level_idx=i,
            dtype=dtype,
            device=device,
            with_stride=with_stride)
        multi_level_priors.append(priors)
    return multi_level_priors

def single_level_grid_priors(featmap_size: Tuple[int],
                            level_idx: int,
                            dtype: torch.dtype = torch.float32,
                            device='cuda',
                            with_stride: bool = False):
    """Generate grid Points of a single level.

    Note:
        This function is usually called by method ``self.grid_priors``.

    Args:
        featmap_size (tuple[int]): Size of the feature maps, arrange as
            (h, w).
        level_idx (int): The index of corresponding feature map level.
        dtype (:obj:`dtype`): Dtype of priors. Defaults to torch.float32.
        device (str | torch.device): The device the tensor will be put on.
            Defaults to 'cuda'.
        with_stride (bool): Concatenate the stride to the last dimension
            of points.

    Return:
        Tensor: Points of single feature levels.
        The shape of tensor should be (N, 2) when with stride is
        ``False``, where N = width * height, width and height
        are the sizes of the corresponding feature level,
        and the last dimension 2 represent (coord_x, coord_y),
        otherwise the shape should be (N, 4),
        and the last dimension 4 represent
        (coord_x, coord_y, stride_w, stride_h).
    """
    feat_h, feat_w = featmap_size
    offset = 0.5
    strides=[8, 16, 32]
    strides = [_pair(stride) for stride in strides]
    stride_w, stride_h = strides[level_idx]
    shift_x = (torch.arange(0, feat_w, device=device) +
                offset) * stride_w
    # keep featmap_size as Tensor instead of int, so that we
    # can convert to ONNX correctly
    shift_x = shift_x.to(dtype)

    shift_y = (torch.arange(0, feat_h, device=device) +
                offset) * stride_h
    # keep featmap_size as Tensor instead of int, so that we
    # can convert to ONNX correctly
    shift_y = shift_y.to(dtype)
    shift_xx, shift_yy = _meshgrid(shift_x, shift_y)
    if not with_stride:
        shifts = torch.stack([shift_xx, shift_yy], dim=-1)
    else:
        # use `shape[0]` instead of `len(shift_xx)` for ONNX export
        stride_w = shift_xx.new_full((shift_xx.shape[0], ),
                                        stride_w).to(dtype)
        stride_h = shift_xx.new_full((shift_yy.shape[0], ),
                                        stride_h).to(dtype)
        shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h],
                                dim=-1)
    all_points = shifts.to(device)
    return all_points


def distance2bbox(
    points,
    distance,
    max_shape=None
):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (B, N, 2) or (N, 2).
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom). Shape (B, N, 4) or (N, 4)
        max_shape (Union[Sequence[int], Tensor, Sequence[Sequence[int]]],
            optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If priors shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.

    Returns:
        Tensor: Boxes with shape (N, 4) or (B, N, 4)
    """

    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    bboxes = torch.stack([x1, y1, x2, y2], -1)
    if max_shape is not None:
        if bboxes.dim() == 2 and not torch.onnx.is_in_onnx_export():
            # speed up
            bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
            bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
            return bboxes

        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape)
        max_shape = max_shape[..., :2].type_as(x1)
        if max_shape.ndim == 2:
            assert bboxes.ndim == 3
            assert max_shape.size(0) == bboxes.size(0)

        min_xy = x1.new_tensor(0)
        max_xy = torch.cat([max_shape, max_shape],
                           dim=-1).flip(-1).unsqueeze(-2)
        bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    return bboxes


def decode(
    points: torch.Tensor,
    pred_bboxes: torch.Tensor,
    stride: torch.Tensor,
    max_shape=None
) -> torch.Tensor:
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (B, N, 2) or (N, 2).
        pred_bboxes (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom). Shape (B, N, 4)
            or (N, 4)
        stride (Tensor): Featmap stride.
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If priors shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]],
            and the length of max_shape should also be B.
            Default None.
    Returns:
        Tensor: Boxes with shape (N, 4) or (B, N, 4)
    """
    assert points.size(-2) == pred_bboxes.size(-2)
    assert points.size(-1) == 2
    assert pred_bboxes.size(-1) == 4

    pred_bboxes = pred_bboxes * stride[None, :, None]

    return distance2bbox(points, pred_bboxes, max_shape)


def filter_scores_and_topk(scores, score_thr, topk, results=None):
    """Filter results using score threshold and topk candidates.

    Args:
        scores (Tensor): The scores, shape (num_bboxes, K).
        score_thr (float): The score filter threshold.
        topk (int): The number of topk candidates.
        results (dict or list or Tensor, Optional): The results to
           which the filtering rule is to be applied. The shape
           of each item is (num_bboxes, N).

    Returns:
        tuple: Filtered results

            - scores (Tensor): The scores after being filtered, \
                shape (num_bboxes_filtered, ).
            - labels (Tensor): The class labels, shape \
                (num_bboxes_filtered, ).
            - anchor_idxs (Tensor): The anchor indexes, shape \
                (num_bboxes_filtered, ).
            - filtered_results (dict or list or Tensor, Optional): \
                The filtered results. The shape of each item is \
                (num_bboxes_filtered, N).
    """
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)

    num_topk = min(topk, valid_idxs.size(0))
    # torch.sort is actually faster than .topk (at least on GPUs)
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)

    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError(f'Only supports dict or list or Tensor, '
                                      f'but get {type(results)}.')
    return scores, labels, keep_idxs, filtered_results


class NMSop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bboxes: Tensor, scores: Tensor, iou_threshold: float,
                offset: int, score_threshold: float, max_num: int) -> Tensor:
        is_filtering_by_score = score_threshold > 0
        if is_filtering_by_score:
            valid_mask = scores > score_threshold
            bboxes, scores = bboxes[valid_mask], scores[valid_mask]
            valid_inds = torch.nonzero(
                valid_mask, as_tuple=False).squeeze(dim=1)

        def load_ext(name, funcs):
            import importlib
            ext = importlib.import_module('mmcv.' + name)
            for fun in funcs:
                assert hasattr(ext, fun), f'{fun} miss in module {name}'
            return ext
        ext_module = load_ext('_ext', ['nms', 'softnms', 'nms_match', 'nms_rotated', 'nms_quadri'])
        inds = ext_module.nms(
            bboxes, scores, iou_threshold=float(iou_threshold), offset=offset)

        if max_num > 0:
            inds = inds[:max_num]
        if is_filtering_by_score:
            inds = valid_inds[inds]
        return inds

def nms(boxes,
        scores,
        iou_threshold: float,
        offset: int = 0,
        score_threshold: float = 0,
        max_num: int = -1):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either torch tensor or numpy array. GPU NMS will be used
    if the input is gpu tensor, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        boxes (torch.Tensor or np.ndarray): boxes in shape (N, 4).
        scores (torch.Tensor or np.ndarray): scores in shape (N, ).
        iou_threshold (float): IoU threshold for NMS.
        offset (int, 0 or 1): boxes' width or height is (x2 - x1 + offset).
        score_threshold (float): score threshold for NMS.
        max_num (int): maximum number of boxes after NMS.

    Returns:
        tuple: kept dets (boxes and scores) and indice, which always have
        the same data type as the input.

    Example:
        >>> boxes = np.array([[49.1, 32.4, 51.0, 35.9],
        >>>                   [49.3, 32.9, 51.0, 35.3],
        >>>                   [49.2, 31.8, 51.0, 35.4],
        >>>                   [35.1, 11.5, 39.1, 15.7],
        >>>                   [35.6, 11.8, 39.3, 14.2],
        >>>                   [35.3, 11.5, 39.9, 14.5],
        >>>                   [35.2, 11.7, 39.7, 15.7]], dtype=np.float32)
        >>> scores = np.array([0.9, 0.9, 0.5, 0.5, 0.5, 0.4, 0.3],\
               dtype=np.float32)
        >>> iou_threshold = 0.6
        >>> dets, inds = nms(boxes, scores, iou_threshold)
        >>> assert len(inds) == len(dets) == 3
    """
    is_numpy = False
    if isinstance(boxes, np.ndarray):
        is_numpy = True
        boxes = torch.from_numpy(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    assert boxes.size(1) == 4
    assert boxes.size(0) == scores.size(0)
    assert offset in (0, 1)

    inds = NMSop.apply(boxes, scores, iou_threshold, offset, score_threshold,
                       max_num)
    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
    if is_numpy:
        dets = dets.cpu().numpy()
        inds = inds.cpu().numpy()
    return dets, inds


def batched_nms(boxes,
                scores,
                idxs,
                nms_cfg,
                class_agnostic: bool = False):
    r"""Performs non-maximum suppression in a batched fashion.

    Modified from `torchvision/ops/boxes.py#L39
    <https://github.com/pytorch/vision/blob/
    505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39>`_.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Note:
        In v1.4.1 and later, ``batched_nms`` supports skipping the NMS and
        returns sorted raw results when `nms_cfg` is None.

    Args:
        boxes (torch.Tensor): boxes in shape (N, 4) or (N, 5).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict | optional): Supports skipping the nms when `nms_cfg`
            is None, otherwise it should specify nms type and other
            parameters like `iou_thr`. Possible keys includes the following.

            - iou_threshold (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
              number of boxes is large (e.g., 200k). To avoid OOM during
              training, the users could set `split_thr` to a small value.
              If the number of boxes is greater than the threshold, it will
              perform NMS on each group of boxes separately and sequentially.
              Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class. Defaults to False.

    Returns:
        tuple: kept dets and indice.

        - boxes (Tensor): Bboxes with score after nms, has shape
          (num_bboxes, 5). last dimension 5 arrange as
          (x1, y1, x2, y2, score)
        - keep (Tensor): The indices of remaining boxes in input
          boxes.
    """
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return torch.cat([boxes, scores[:, None]], -1), inds

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        # When using rotated boxes, only apply offsets on center.
        if boxes.size(-1) == 5:
            # Strictly, the maximum coordinates of the rotating box
            # (x,y,w,h,a) should be calculated by polygon coordinates.
            # But the conversion from rotated box to polygon will
            # slow down the speed.
            # So we use max(x,y) + max(w,h) as max coordinate
            # which is larger than polygon max coordinate
            # max(x1, y1, x2, y2,x3, y3, x4, y4)
            max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
            offsets = idxs.to(boxes) * (
                max_coordinate + torch.tensor(1).to(boxes))
            boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
            boxes_for_nms = torch.cat([boxes_ctr_for_nms, boxes[..., 2:5]],
                                      dim=-1)
        else:
            max_coordinate = boxes.max()
            offsets = idxs.to(boxes) * (
                max_coordinate + torch.tensor(1).to(boxes))
            boxes_for_nms = boxes + offsets[:, None]
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr:
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]

        # This assumes `dets` has arbitrary dimensions where
        # the last dimension is score.
        # Currently it supports bounding boxes [x1, y1, x2, y2, score] or
        # rotated boxes [cx, cy, w, h, angle_radian, score].

        scores = dets[:, -1]
    else:
        max_num = nms_cfg_.pop('max_num', -1)
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        # Some type of nms would reweight the score, such as SoftNMS
        scores_after_nms = scores.new_zeros(scores.size())
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero(as_tuple=False).view(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    return boxes, keep


def bbox_post_process(results,
                      rescale: bool = False,
                      with_nms: bool = True,
                      cfg_nms={'type': 'nms', 'iou_threshold': 0.5},
                      img_meta=None):
    """bbox post-processing method.

    The boxes would be rescaled to the original image scale and do
    the nms operation. Usually `with_nms` is False is used for aug test.

    Args:
        results (:obj:`InstaceData`): Detection instance results,
            each item has shape (num_bboxes, ).
        cfg (ConfigDict): Test / postprocessing configuration,
            if None, test_cfg would be used.
        rescale (bool): If True, return boxes in original image space.
            Default to False.
        with_nms (bool): If True, do nms before return boxes.
            Default to True.
        img_meta (dict, optional): Image meta info. Defaults to None.

    Returns:
        :obj:`InstanceData`: Detection results of each image
        after the post process.
        Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
                (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
                the last dimension 4 arrange as (x1, y1, x2, y2).
    """
    if hasattr(results, 'score_factors'):
        # TODO： Add sqrt operation in order to be consistent with
        #  the paper.
        score_factors = results.pop('score_factors')
        results.scores = results.scores * score_factors

    # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
    if with_nms and results.bboxes.numel() > 0:
        bboxes = results.bboxes
        det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                            results.labels, cfg_nms)
        results = results[keep_idxs]
        # some nms would reweight the score, such as softnms
        results.scores = det_bboxes[:, -1]
        # results = results[:cfg.max_per_img]

    return results


def post_process(outputs, ori_shape, input_shape, with_mmcv=False):
    # out = torch.Tensor(np.concatenate([outputs[1], np.ones((1,8064,1)), torch.tensor(outputs[0]).sigmoid()], axis=2))
    # det = non_max_suppression(out, 0.1, 0.5)[0]
    # print(det)
    flatten_cls_scores = torch.from_numpy(outputs[0]).sigmoid().cuda()
    flatten_bbox_preds = torch.from_numpy(outputs[1]).cuda()
    # featmap_sizes = [torch.Size([64, 96]), torch.Size([32, 48]), torch.Size([16, 24])]
    # featmap_sizes = [torch.Size([80, 80]), torch.Size([40, 40]), torch.Size([20, 20])]
    featmap_sizes = [torch.Size([24, 24]), torch.Size([12, 12]), torch.Size([6, 6])]  # TODO
    mlvl_priors = grid_priors(
                    featmap_sizes,
                    dtype=torch.float32,
                    device='cuda')
    flatten_priors = torch.cat(mlvl_priors)
    num_base_priors = 1
    featmap_strides = [8, 16, 32]
    mlvl_strides = [
        flatten_priors.new_full(
            (featmap_size.numel() * num_base_priors, ), stride) for
        featmap_size, stride in zip(featmap_sizes, featmap_strides)
    ]
    flatten_stride = torch.cat(mlvl_strides)
    flatten_decoded_bboxes = decode(flatten_priors[None], flatten_bbox_preds, flatten_stride)
    flatten_objectness = [None for _ in range(1)]

    results_list = []
    for (bboxes, scores, objectness) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                            flatten_objectness):
        if not with_mmcv:
            flatten_decoded_bboxes = flatten_decoded_bboxes.cpu().numpy()
            flatten_cls_scores = flatten_cls_scores.cpu().numpy()
            out = torch.Tensor(np.concatenate([flatten_decoded_bboxes, np.ones((1,756,1)), flatten_cls_scores], axis=2))  # TODO 8400
            det = non_max_suppression(out, 0.5, 0.5)[0]
            return det
        
        scale_factor = (input_shape[0]/ori_shape[1], input_shape[1]/ori_shape[0])
        
        pad_param = None
        score_thr = 0.0001

        if scores.shape[0] == 0:
            empty_results = InstanceData()
            empty_results.bboxes = bboxes
            empty_results.scores = scores[:, 0]
            empty_results.labels = scores[:, 0].int()
            results_list.append(empty_results)
            continue

        nms_pre = 10000
        multi_label = True
        if multi_label is False:
            scores, labels = scores.max(1, keepdim=True)
            scores, _, keep_idxs, results = filter_scores_and_topk(
                scores,
                score_thr,
                nms_pre,
                results=dict(labels=labels[:, 0]))
            labels = results['labels']
        else:
            scores, labels, keep_idxs, _ = filter_scores_and_topk(
                scores, score_thr, nms_pre)
        results = InstanceData(
            scores=scores, labels=labels, bboxes=bboxes[keep_idxs])
        rescale = True
        if rescale:
            if pad_param is not None:
                results.bboxes -= results.bboxes.new_tensor([
                    pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                ])
            results.bboxes /= results.bboxes.new_tensor(
                scale_factor).repeat((1, 2))
        cfg_nms = {'type': 'nms', 'iou_threshold': 0.5}
        results = bbox_post_process(
            results=results,
            rescale=False,
            with_nms=True,
            cfg_nms=cfg_nms)
        results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
        results.bboxes[:, 1::2].clamp_(0, ori_shape[0])
        results_list.append(results)
    result = results_list[0]
    return result


def expand(bbox, w, h, ratio=0.2):
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    bbox[0] = max(int(bbox[0] - bbox_w * ratio), 0)
    bbox[1] = max(int(bbox[1] - bbox_h * ratio), 0)
    bbox[2] = min(int(bbox[2] + bbox_w * ratio), w)
    bbox[3] = min(int(bbox[3] + bbox_h * ratio), h)
    return bbox


def head_onnx_infer(sess, img):
    # img = cv2.imread(img_path)
    ori_img = img.copy()
    ori_shape = img.shape
    input_shape = (192, 192)
    img = cv2.resize(img, input_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    img = img.transpose(2, 0, 1).reshape(1, 3, *input_shape)

    # onnx_path= '/home/aizhiqing/data1/aizhiqing/code_python/gaze_estimation/6DRepNet-1/weights/dms_face_hand.onnx'
    # sess = ort.InferenceSession(onnx_path,providers=['CUDAExecutionProvider']) # 'CPUExecutionProvider'
    input_name = sess.get_inputs()[0].name
    output_name = [output.name for output in sess.get_outputs()]
    outputs = sess.run(output_name, {input_name:img})
    
    with_mmcv = False
    result = post_process(outputs, ori_shape, input_shape, with_mmcv=with_mmcv) #! TODO
    # class_names = ['person', 'car', 'bus', 'truck', 'motor','head','sign','road_arrow','traffic_cone','traffic_light','rear']
    class_names = ['face', 'head', 'hand']
    
    head_list = []
    h_factor = ori_shape[1] / input_shape[0]
    w_factor = ori_shape[0] / input_shape[1]
    for bbox in result:
        label_idx = int(bbox[-1])
        label = class_names[label_idx]
        score = '{:.2f}'.format(bbox[-2])
        bbox[0] *= h_factor
        bbox[1] *= w_factor
        bbox[2] *= h_factor
        bbox[3] *= w_factor
        
        if label_idx ==1:
            head_list.append(bbox[:4].cpu().numpy())
        # img_vis=cv2.rectangle(img_vis, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=plattes[label_idx], thickness=2) 
        # img_vis=cv2.putText(img_vis, score, (int(bbox[0]), int(bbox[1]-3)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
        #             color=plattes[label_idx], thickness=2)
    return head_list

if __name__ == '__main__':
    plattes = [(0,0,0),(0,0,255),(255,0,0),(0,255,0),(255,255,0),(255,0,255),(0,255,255),(255,255,255),(100,255,0),(100,0,255),(255,100,0),(0,100,255),(255,0,100),(0,255,100)]
    
    img_path = '/home/aizhiqing/data1/aizhiqing/input_dir/hand_cls_testdata/20240318_DMS_imgs/20240313175154_000278/20240313175154_000278_20.jpg'
    # input_shape = (640, 640)
    input_shape = (192, 192)
    img = cv2.imread(img_path)
    ori_img = img.copy()
    ori_shape = img.shape
    img = cv2.resize(img, input_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    img = img.transpose(2, 0, 1).reshape(1, 3, *input_shape)

    onnx_path= '/home/aizhiqing/data1/aizhiqing/code_python/gaze_estimation/6DRepNet-1/weights/dms_face_hand.onnx'
    sess = ort.InferenceSession(onnx_path,providers=['CUDAExecutionProvider']) # 'CPUExecutionProvider'
    input_name = sess.get_inputs()[0].name
    output_name = [output.name for output in sess.get_outputs()]
    outputs = sess.run(output_name, {input_name:img})

    with_mmcv = False
    
    result = post_process(outputs, ori_shape, input_shape, with_mmcv=with_mmcv) #! TODO
    
    # class_names = ['person', 'car', 'bus', 'truck', 'motor','head','sign','road_arrow','traffic_cone','traffic_light','rear']
    class_names = ['face', 'head', 'hand']
    
    head_list = []
    face_bbox = 0
    area = 0
    has_face_bbox = False
    img_vis = ori_img.copy()
    if with_mmcv:
        bboxes = result.bboxes.cpu().numpy()
        scores = result.scores.cpu().numpy()
        labels = result.labels.cpu().numpy()
        valid_bboxes_mask= scores > 0.5
        valid_bboxes = bboxes[valid_bboxes_mask]
        valid_labels = labels[valid_bboxes_mask]
        for idx, bbox in enumerate(valid_bboxes):
            label_idx = int(valid_labels[idx])
            label = class_names[label_idx]
            score = '{:.2f}'.format(scores[idx])
            img=cv2.rectangle(ori_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=plattes[label_idx], thickness=2) 
            img=cv2.putText(img, score, (int(bbox[0]), int(bbox[1]-3)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
                        color=plattes[label_idx], thickness=2)
        # cv2.imwrite('/home/data/wanghonggang/whg_test/test.jpg', img)
    else:
        h_factor = ori_shape[1] / input_shape[0]
        w_factor = ori_shape[0] / input_shape[1]
        for bbox in result:
            label_idx = int(bbox[-1])
            
            label = class_names[label_idx]
            score = '{:.2f}'.format(bbox[-2])
            bbox[0] *= h_factor
            bbox[1] *= w_factor
            bbox[2] *= h_factor
            bbox[3] *= w_factor
            
            if label_idx == 0:
                has_face_bbox = True
                bbox_area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
                if bbox_area > area:
                    area = bbox_area
                    face_bbox = bbox
            img_vis=cv2.rectangle(img_vis, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=plattes[label_idx], thickness=2) 
            img_vis=cv2.putText(img_vis, score, (int(bbox[0]), int(bbox[1]-3)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
                        color=plattes[label_idx], thickness=2)
        # cv2.imwrite('/home/data/wanghonggang/whg_test/test.jpg', img_vis)
        
    # if has_face_bbox:
    #     class_names=('close_eyes', 'yawn', 'phone', 'cigarette', 'squint', 'sunglass')
        
    #     onnx_path= '/home/data/wanghonggang/whg_test/dms_face_action.onnx'
    #     sess = ort.InferenceSession(onnx_path,providers=['CUDAExecutionProvider']) # 'CPUExecutionProvider'
    #     input_name = sess.get_inputs()[0].name
    #     output_name = [output.name for output in sess.get_outputs()]
        
    #     face_bbox = expand(face_bbox, 1280, 720, ratio=0.2)
    #     input_frame = ori_img[int(face_bbox[1]):int(face_bbox[3]), int(face_bbox[0]):int(face_bbox[2]), :]
    #     ori_shape = input_frame.shape
    #     img = cv2.resize(input_frame, input_shape)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = img.astype(np.float32)/255.
    #     img = img.transpose(2, 0, 1).reshape(1, 3, *input_shape)
    #     outputs = sess.run(output_name, {input_name:img})
    #     with_mmcv = False
    #     result = post_process(outputs, ori_shape, input_shape, with_mmcv=with_mmcv)
    #     if not with_mmcv:
    #         h_factor = ori_shape[1] / input_shape[0]
    #     w_factor = ori_shape[0] / input_shape[1]
    #     for bbox in result:
    #         label_idx = int(bbox[-1])
            
    #         label = class_names[label_idx]
    #         score = '{:.2f}'.format(bbox[-2])
    #         bbox[0] *= h_factor
    #         bbox[1] *= w_factor
    #         bbox[2] *= h_factor
    #         bbox[3] *= w_factor
            
    #         if label_idx == 0:
    #             has_face_bbox = True
    #             bbox_area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
    #             if bbox_area > area:
    #                 area = bbox_area
    #                 face_bbox = bbox
    #         img_vis=cv2.rectangle(img_vis, (int(bbox[0]+face_bbox[0]), int(bbox[1]+face_bbox[1])), (int(bbox[2]+face_bbox[0]), int(bbox[3]+face_bbox[1])), color=plattes[label_idx], thickness=2) 
    #         img_vis=cv2.putText(img_vis, score, (int(bbox[0]+face_bbox[0]), int(bbox[1]+face_bbox[1]-3)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
    #                     color=plattes[label_idx], thickness=2)
        # cv2.imwrite('/home/data/wanghonggang/whg_test/test2.jpg', img_vis)