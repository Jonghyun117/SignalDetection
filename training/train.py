# training/train.py
import argparse, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import AMCNet
from dataset import AMCDataset, _preprocess
from simulate import MODULATIONS
import numpy as np


def _eval_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += len(y)
    return correct / total


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds     = AMCDataset(n_per_class_per_snr=args.n_per, augment=True)
    n_val  = int(0.15 * len(ds))
    tr_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val],
                                 generator=torch.Generator().manual_seed(42))
    tr_loader  = DataLoader(tr_ds,  batch_size=args.batch, shuffle=True,  num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model     = AMCNet(num_classes=len(MODULATIONS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = correct = total = 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct  += (logits.argmax(1) == y).sum().item()
            total    += len(y)
        scheduler.step()
        val_acc = _eval_accuracy(model, val_loader, device)
        print(f"Epoch {epoch:3d} | loss={loss_sum/total:.4f} "
              f"| train={correct/total:.3f} | val={val_acc:.3f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), args.save)
    print(f"Best val_acc: {best_val:.3f} -> saved to {args.save}")


def adapt_bn(model_path, real_iq_list, output_path):
    """Update only BatchNorm running statistics using real SG data.

    real_iq_list: list of (i_arr, q_arr) float32 numpy arrays from the real SG.
    All model weights are frozen; only BN running_mean/running_var are updated.
    Typically 200-500 samples per class are sufficient.
    """
    model = AMCNet(num_classes=len(MODULATIONS))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.train()
            m.reset_running_stats()
        else:
            for p in m.parameters():
                p.requires_grad_(False)

    with torch.no_grad():
        for i_arr, q_arr in real_iq_list:
            feat = _preprocess(i_arr, q_arr)
            model(torch.from_numpy(feat).unsqueeze(0))

    torch.save(model.state_dict(), output_path)
    print(f"BN-adapted model saved -> {output_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int,   default=50)
    p.add_argument('--batch',  type=int,   default=256)
    p.add_argument('--lr',     type=float, default=1e-3)
    p.add_argument('--n_per',  type=int,   default=500)
    p.add_argument('--save',   default='model/best.pth')
    args = p.parse_args()
    os.makedirs('model', exist_ok=True)
    train(args)
