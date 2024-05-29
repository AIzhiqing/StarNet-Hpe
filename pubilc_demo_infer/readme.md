# 2.1
## Performance (less than WHENet and 6DRepNet)
Input shape: (3, 244, 244)
Flops: 1.75 GFLOPs
Params: 7.04 M

## Error(degree) (beyond WHENet and 6DRepNet)
- v2.1: Yaw: 2.5130, Pitch: 3.1318, Roll: 2.1042, MAE: 2.5830
- v2.2: Yaw: 2.3148, Pitch: 3.1879, Roll: 2.1651, MAE: 2.5559

## simulation
9331: 1.24ms, mac ratio: 67.05%

# 2.3
## Performance (less than WHENet and 6DRepNet)
Input shape: (3, 244, 244)
Flops: 1.66 GFLOPs
Params: 5.7 M

## Error(degree) (beyond WHENet and 6DRepNet)
- v2.3: Yaw: , Pitch: , Roll: , MAE: 

## latency
530: 6ms



BIWI
Yaw: 0.6116, Pitch: 0.6408, Roll: 0.7055, MAE: 0.6527

train: 300W_LP+BIWI, test: AFLW2000
SixDRepNet_1705557943_bs128/_epoch_11.pt
Yaw: 2.2129, Pitch: 3.0976, Roll: 2.1689, MAE: 2.4931
SixDRepNet_1705557943_bs128/_epoch_21.pt
Yaw: 2.4811, Pitch: 3.4407, Roll: 2.3547, MAE: 2.7588
SixDRepNet_1705557943_bs128/_epoch_31.pt
Yaw: 2.5476, Pitch: 3.4815, Roll: 2.3562, MAE: 2.7951
SixDRepNet_1705557943_bs128/_epoch_51.pt
Yaw: 2.5053, Pitch: 3.4304, Roll: 2.3481, MAE: 2.7613
SixDRepNet_1705557943_bs128/_epoch_61.pt
Yaw: 2.5180, Pitch: 3.4173, Roll: 2.3548, MAE: 2.7634
SixDRepNet_1705557943_bs128/_epoch_71.pt
Yaw: 2.5200, Pitch: 3.4551, Roll: 2.3621, MAE: 2.7791
SixDRepNet_1705557943_bs128/_epoch_81.pt
Yaw: 2.5658, Pitch: 3.4061, Roll: 2.3325, MAE: 2.7681
SixDRepNet_1705557943_bs128/_epoch_91.pt
Yaw: 2.5653, Pitch: 3.4333, Roll: 2.3606, MAE: 2.7864
SixDRepNet_1705557943_bs128/_epoch_100.pt
Yaw: 2.5889, Pitch: 3.5013, Roll: 2.3778, MAE: 2.8227



SixDRepNet_1706106653_bs128 + data + 360



