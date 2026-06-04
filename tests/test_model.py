# tests/test_model.py
import torch
import sys
sys.path.insert(0, 'training')
from model import AMCNet
from simulate import MODULATIONS

N = len(MODULATIONS)  # 21

def test_forward_shape():
    model = AMCNet(num_classes=N)
    out = model(torch.randn(4, 3, 1024))
    assert out.shape == (4, N), f"Expected (4,{N}), got {out.shape}"

def test_param_count():
    model = AMCNet(num_classes=N)
    n = sum(p.numel() for p in model.parameters())
    assert n < 500_000, f"Too many params: {n}"

def test_eval_deterministic():
    model = AMCNet(num_classes=N)
    model.eval()
    x = torch.randn(1, 3, 1024)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2), "Eval mode should be deterministic"
