import argparse
from typing import Dict, Any
import collections
import os

import numpy as np

import torch

from flax.serialization import from_bytes


def tree_iter(root: Dict[str, Any], pre_fix: str, return_dict: Dict[str, torch.Tensor]):
    if isinstance(root, np.ndarray):
        return_dict[pre_fix] = torch.from_numpy(root)
        return

    for k, v in root.items():
        tree_iter(v, f"{pre_fix}.{k}", return_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="ckpt/small_1000k.bin")
    parser.add_argument("--save_fp", type=str, default="ckpt/small_1000k.pth")
    args = parser.parse_args()
    save_path = os.path.dirname(args.save_fp)
    os.makedirs(save_path, exist_ok=True)

    with open(args.ckpt, "rb") as state_f:
        state = from_bytes(None, state_f.read())

    return_dict: Dict[str, torch.Tensor] = collections.OrderedDict()
    tree_iter(state, "module", return_dict)

    torch.save(return_dict, args.save_fp)
