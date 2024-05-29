import os


# gt file: 分神次数
"""
P: 正样本数，即gt files 中的分神标签次数；
TP: 正确的报警次数，真正例；
FP: 错误的报警次数，假正例；
Recall = TP/ P;
Precision  = TP/ (TP+ FP);
"""
dict ={
    'Normal_20240130103231_01': {'count': 0, 'frame_id': []},
    'Normal_20240130102331_01': {'count': 6, 'frame_id': [72, 1080, 1460, 1560, 2382, 4502]},
    'Normal_20240130102030_0': {'count': 3, 'frame_id': [ 1590, 1700, ]},
    'Normal_20240130101430_01': {'count': 13, 'frame_id': [65, 240, 515, 600, 720, 930, 2189, 3438, 3736, 4102, 4314, 4409]},
    'Normal_20240130101730_01': {'count': 2, 'frame_id': [62, 1237, ]},
    'Normal_20240130104731_01': {'count': 3, 'frame_id': [4129, 4380]},
}


if __name__ == '__main__':
    
    import numpy as np
    from scipy.spatial.transform import Rotation

    def angle_between_vectors(vec1, vec2):
        """
        计算两个向量之间的角度
        """
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        cos_theta = dot_product / norm_product
        # 将cos值转换为角度并确保范围在[0, pi]
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return theta

    def maev_loss(pred_rotations, gt_rotations):
        """
        MAEV损失函数的实现
        pred_rotations: 预测的头部姿态旋转矩阵列表，形状为 (batch_size, 3, 3)
        gt_rotations: 真实的头部姿态旋转矩阵列表，对应每个预测，形状同上
        """
        assert pred_rotations.shape == gt_rotations.shape
        
        # 将旋转矩阵转换为前向方向向量
        pred_directions = Rotation.from_matrix(pred_rotations[:, :3, 2]).as_rotvec()  # 取第三列作为看向正前方的方向向量
        gt_directions = Rotation.from_matrix(gt_rotations[:, :3, 2]).as_rotvec()

        # 计算每一对预测和真实向量的角度差
        angle_errors = np.array([angle_between_vectors(pred_dir, gt_dir) for pred_dir, gt_dir in zip(pred_directions, gt_directions)])

        # 计算平均角度误差
        mean_angle_error = np.mean(angle_errors)

        return mean_angle_error

    # 示例使用
    pred_rots = np.random.rand(1, 3, 3)  # 假设是随机生成的旋转矩阵
    gt_rots = np.random.rand(1, 3, 3)  # 假设是对应的真值旋转矩阵

    loss = maev_loss(pred_rots, gt_rots)
    print(f"MAEV Loss: {loss:.4f} radians")

    # 实际应用中，请确保预处理旋转矩阵以确保它们都是有效的三维旋转矩阵