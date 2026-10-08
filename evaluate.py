#!/usr/bin/env python
"""Evaluate trained H2LO checkpoints on the ds006557 test splits with the paper's eight metrics.

Per fold, the metrics are averaged over the test subjects; across folds the table reports
mean +- std (population std) of the fold means.

    python evaluate.py --data_root /path/to/prepared --contrast T1w --folds 0 1 2 3 4 \
        --ckpt "runs/{contrast}_fold{fold}/best_model.pth" --out results_T1w.csv
"""
import argparse
import csv

import numpy as np
import torch

from h2lo.data import make_splits
from h2lo.metrics import METRICS, all_metrics
from h2lo.model import H2LO, predict_full_volume


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck.get('args', {})
    model = H2LO(a.get('feature_dim', 128), a.get('encoder_base_channels', 32), a.get('trunk_type', 'siren'),
                 a.get('decoder_depth', 4), a.get('decoder_width', 256), a.get('omega', 30.)).to(device)
    model.load_state_dict(ck['model_state_dict'])
    return model.eval(), ck.get('epoch', -1)


def main():
    p = argparse.ArgumentParser(description="Evaluate H2LO on ds006557")
    p.add_argument('--data_root', required=True)
    p.add_argument('--contrast', default='T1w', choices=['T1w', 'T2w'])
    p.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--ckpt', required=True, help='checkpoint path; may contain {contrast} and {fold}')
    p.add_argument('--subjects', nargs='*', default=None, help='restrict to these test subjects')
    p.add_argument('--chunk_size', type=int, default=100000)
    p.add_argument('--no_amp', dest='amp', action='store_false')
    p.add_argument('--out', default=None, help='per-subject CSV')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows, fold_means = [], []
    for fold in args.folds:
        model, epoch = load_model(args.ckpt.format(contrast=args.contrast, fold=fold), device)
        test = make_splits(args.data_root, args.contrast, fold)['test']
        fold_rows = []
        for i in range(len(test)):
            v = test[i]
            if args.subjects and v['subject'] not in args.subjects:
                continue
            hf = torch.from_numpy(v['hf'])[None, None].to(device)
            pred = predict_full_volume(model, hf, args.chunk_size, use_amp=args.amp).cpu().numpy()
            r = {'contrast': args.contrast, 'fold': fold, 'epoch': epoch, 'subject': v['subject'],
                 **all_metrics(pred, v['lf'])}
            print(' '.join(f"{k}={x:.4f}" if isinstance(x, float) else f"{k}={x}" for k, x in r.items()), flush=True)
            fold_rows.append(r)
        rows += fold_rows
        fold_means.append({k: np.mean([r[k] for r in fold_rows]) for k in METRICS})

    print(f"\n{args.contrast}, {len(fold_means)} fold(s): mean +- std over fold means")
    for k in METRICS:
        x = np.array([f[k] for f in fold_means])
        print(f"  {k:14s} {x.mean():.4f} +- {x.std():.4f}")
    if args.out:
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == '__main__':
    main()
