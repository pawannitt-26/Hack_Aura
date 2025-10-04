EPOCHS = 10
MOSAIC = 0.4
OPTIMIZER = 'AdamW'
MOMENTUM = 0.9
LR0 = 0.0001
LRF = 0.0001
SINGLE_CLS = False
DEVICE = 'auto'  # 'auto' selects MPS (Apple GPU) if available, else CUDA, else CPU
import argparse
from ultralytics import YOLO
import os
import sys
import torch

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    # epochs
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    # mosaic
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation')
    # optimizer
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer')
    # momentum
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum')
    # lr0
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    # lrf
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    # single_cls
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Single class training')
    # device
    parser.add_argument('--device', type=str, default=DEVICE, help='Device to use: auto, mps, cuda, cpu, or index (0,1,...)')
    args = parser.parse_args()
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    model = YOLO(os.path.join(this_dir, "models", "pretrained_yolov8s.pt"))
    # Resolve device
    if args.device == 'auto':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            resolved_device = 'mps'
        elif torch.cuda.is_available():
            resolved_device = 'cuda'
        else:
            resolved_device = 'cpu'
    else:
        resolved_device = args.device

    print(f"Using device: {resolved_device}")
    results = model.train(
        data=os.path.join(this_dir, "configs", "dataset_config.yaml"), 
        epochs=args.epochs,
        device=resolved_device,
        single_cls=args.single_cls, 
        mosaic=args.mosaic,
        optimizer=args.optimizer, 
        lr0 = args.lr0, 
        lrf = args.lrf, 
        momentum=args.momentum,
        project=os.path.join(this_dir, "runs"),
        name="safety_equipment_training"
    )
'''
Mixup boost val pred but reduces test pred
Mosaic shouldn't be 1.0  
'''


'''
                   from  n    params  module                                       arguments
  0                  -1  1       928  ultralytics.nn.modules.conv.Conv             [3, 32, 3, 2]
  1                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  2                  -1  1     29056  ultralytics.nn.modules.block.C2f             [64, 64, 1, True]
  3                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  4                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  5                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  6                  -1  2    788480  ultralytics.nn.modules.block.C2f             [256, 256, 2, True]
  7                  -1  1   1180672  ultralytics.nn.modules.conv.Conv             [256, 512, 3, 2]
  8                  -1  1   1838080  ultralytics.nn.modules.block.C2f             [512, 512, 1, True]
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  1    591360  ultralytics.nn.modules.block.C2f             [768, 256, 1]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]
 16                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]
 19                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1   1969152  ultralytics.nn.modules.block.C2f             [768, 512, 1]
 22        [15, 18, 21]  1   2118757  ultralytics.nn.modules.head.Detect           [7, [128, 256, 512]]
Model summary: 129 layers, 11,138,309 parameters, 11,138,293 gradients, 28.7 GFLOPs
'''