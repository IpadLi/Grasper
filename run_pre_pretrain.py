import sys
import torch
import os

from graph_learning.graph_pretrain import graph_pretrain
from args_cfg import get_args

sys.path.append(".")

if __name__ == '__main__':
    device_id = [0, 1, 2, 3, 4, 5, 6, 7]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
    args = get_args()
    args.cuda = torch.cuda.is_available()
    graph_pretrain(args)