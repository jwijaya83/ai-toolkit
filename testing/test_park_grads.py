# Output preservation backwards two losses separately with the first one's grads parked
# on CPU. The result has to match one backward on the summed loss.
# ./venv/bin/python testing/test_park_grads.py
import copy
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
from extensions_built_in.sd_trainer.SDTrainer import park_grads, restore_grads

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
a = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1)).to(device)
b = copy.deepcopy(a)
x, y = torch.randn(8, 4, device=device), torch.randn(8, 4, device=device)

(a(x).pow(2).mean() + a(y).pow(2).mean()).backward()

groups = [{"params": list(b.parameters())}]
b(x).pow(2).mean().backward()
parked = park_grads(groups)
assert parked and all(p.grad is None for p in b.parameters())
assert all(g.device.type == "cpu" for _, g in parked)
b(y).pow(2).mean().backward()
restore_grads(parked)

for pa, pb in zip(a.parameters(), b.parameters()):
    assert pb.grad.device == pa.device
    assert torch.allclose(pa.grad, pb.grad), "split backward differs from backward on the sum"
print(f"ok on {device}")
