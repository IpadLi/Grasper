import sys
import torch
from mappo.mappo_mtl import mappo_mtl
from grasper_mappo.grasper_mappo_mtl import grasper_mappo_mtl
import os
import random
import numpy as np
from args_cfg import get_args

sys.path.append(".")

if __name__ == '__main__':
    device_id = [0, 1, 2, 3, 4, 5, 6, 7]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
    args = get_args()
    args.cuda = torch.cuda.is_available()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    if args.base_rl == 'mappo':
        mappo_mtl(args)
    elif args.base_rl == 'grasper_mappo':
        grasper_mappo_mtl(args)

