#!/usr/bin/env python
"""Train H2LO on one ds006557 cross-validation fold.

Loss = L1 on K random voxels + lambda_grad * 3D Gaussian-gradient loss on a random sub-volume.
The encoder runs once per step; both terms share its feature map. The checkpoint with the best
validation PSNR is kept as best_model.pth and evaluated on the test split at the end.

    python -m h2lo.train --data_root /path/to/prepared --contrast T1w --fold 0 --save_dir runs/T1w_fold0
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from h2lo.data import PointSampleDataset, make_splits
from h2lo.losses import GaussianGradientLoss3D
from h2lo.metrics import psnr, ssim
from h2lo.model import H2LO, make_coord_grid, predict_full_volume


def random_subvolume_coords(shape, size):
    """Coordinates of a random cubic sub-volume (side min(size, dim)) -> [1, N, 3], (Sh, Sw, Sd)."""
    sub = [min(size, n) for n in shape]
    axes = []
    for n, s in zip(shape, sub):
        o = random.randint(0, n - s)
        axes.append(torch.linspace(-1 + 2 * o / max(n - 1, 1), -1 + 2 * (o + s - 1) / max(n - 1, 1), s))
    coord = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)
    return coord.reshape(1, -1, 3), tuple(sub)


def train_epoch(model, loader, optimizer, scaler, grad_loss, args, device):
    model.train()
    l1 = nn.L1Loss()
    total = 0.
    for batch in loader:
        hf, coord, gt, lf = (batch[k].to(device) for k in ("hf", "coord", "gt", "lf"))
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            feat = model.encoder(hf)
            loss_l1 = l1(model.query(feat, coord), gt)
            loss_grad = torch.zeros((), device=device)
            for b in range(hf.shape[0]):
                sv, (sh, sw, sd) = random_subvolume_coords(lf.shape[1:], args.subvol_size)
                sv = sv.to(device)
                pred = model.query(feat[b:b + 1], sv).view(1, 1, sh, sw, sd)
                target = model.sample(lf[b:b + 1].unsqueeze(1), sv).view(1, 1, sh, sw, sd)
                with torch.amp.autocast('cuda', enabled=False):  # fp32: the gradient loss underflows in fp16
                    loss_grad = loss_grad + grad_loss(pred.float(), target.float())
            loss = loss_l1 + args.lambda_grad * loss_grad / hf.shape[0]
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += loss.item()
    return total / len(loader)


def evaluate(model, volumes, device, args):
    scores = []
    for i in range(len(volumes)):
        v = volumes[i]
        hf = torch.from_numpy(v["hf"])[None, None].to(device)
        pred = predict_full_volume(model, hf, args.eval_chunk_size, use_amp=args.amp).cpu().numpy()
        scores.append((psnr(pred, v["lf"]), ssim(pred, v["lf"])))
    return np.mean(scores, axis=0)


def main():
    p = argparse.ArgumentParser(description="Train H2LO (HF -> LF) on ds006557")
    p.add_argument('--data_root', required=True, help='directory holding ds006557_HFC_T1/ and ds006557_HFC_T2/')
    p.add_argument('--contrast', default='T1w', choices=['T1w', 'T2w'])
    p.add_argument('--fold', type=int, default=0, choices=range(5))
    p.add_argument('--save_dir', required=True)
    # model (paper configuration by default)
    p.add_argument('--feature_dim', type=int, default=128, help='number of basis functions P')
    p.add_argument('--encoder_base_channels', type=int, default=32)
    p.add_argument('--trunk_type', default='siren', choices=['siren', 'mlp'], help="'mlp' = w/o SIREN ablation")
    p.add_argument('--decoder_depth', type=int, default=4)
    p.add_argument('--decoder_width', type=int, default=256)
    p.add_argument('--omega', type=float, default=30.)
    # optimisation
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--K', type=int, default=8000, help='voxels sampled per volume per step')
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lambda_grad', type=float, default=10.0, help="0 = w/o gradient loss ablation")
    p.add_argument('--grad_sigma', type=float, default=1.0)
    p.add_argument('--grad_kernel_size', type=int, default=5)
    p.add_argument('--subvol_size', type=int, default=32)
    p.add_argument('--no_amp', dest='amp', action='store_false')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=0)
    # evaluation / checkpoints
    p.add_argument('--eval_every', type=int, default=5)
    p.add_argument('--eval_chunk_size', type=int, default=100000)
    p.add_argument('--checkpoint_every', type=int, default=10)
    p.add_argument('--resume', default=None)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        args.amp = False
    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)

    splits = make_splits(args.data_root, args.contrast, args.fold)
    print({k: len(v) for k, v in splits.items()}, flush=True)
    loader = DataLoader(PointSampleDataset(splits['train'], args.K), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    model = H2LO(args.feature_dim, args.encoder_base_channels, args.trunk_type,
                 args.decoder_depth, args.decoder_width, args.omega).to(device)
    print(f"H2LO parameters: {sum(q.numel() for q in model.parameters()):,}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    grad_loss = GaussianGradientLoss3D(args.grad_sigma, args.grad_kernel_size).to(device)

    start, best = 1, -np.inf
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        start, best = ck['epoch'] + 1, ck.get('val_psnr', -np.inf)
        for _ in range(1, start):
            scheduler.step()

    for epoch in range(start, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, scaler, grad_loss, args, device)
        msg = f"epoch {epoch}/{args.epochs}  loss {loss:.6f}  lr {optimizer.param_groups[0]['lr']:.2e}"
        state = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(), 'args': vars(args)}
        if epoch % args.eval_every == 0 or epoch == 1:
            val_psnr, val_ssim = evaluate(model, splits['val'], device, args)
            msg += f"  val PSNR {val_psnr:.2f} SSIM {val_ssim:.4f}"
            if val_psnr > best:
                best = val_psnr
                torch.save({**state, 'val_psnr': val_psnr, 'val_ssim': val_ssim}, save / 'best_model.pth')
                msg += "  (best)"
        if epoch % args.checkpoint_every == 0:
            torch.save(state, save / f'checkpoint_epoch{epoch}.pth')
        print(msg, flush=True)
        scheduler.step()

    ck = torch.load(save / 'best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    test_psnr, test_ssim = evaluate(model, splits['test'], device, args)
    print(f"best epoch {ck['epoch']}: test PSNR {test_psnr:.2f} dB, SSIM {test_ssim:.4f}")


if __name__ == '__main__':
    main()
