"""The eight synthesis metrics of the paper, computed on max-normalized volumes in [0, 1].

Intensity:  PSNR, SSIM (mean over axial slices along the first axis), NCC.
Histogram:  Wasserstein-1, NMI, histogram NCC, Bhattacharyya distance, Jensen-Shannon divergence.
"""
import numpy as np
from skimage.metrics import structural_similarity

RANGE = (0.0, 1.0)


def psnr(pred, gt, data_range=1.0):
    mse = float(np.mean((np.asarray(pred, np.float64) - np.asarray(gt, np.float64)) ** 2))
    return float('inf') if mse == 0 else float(20 * np.log10(data_range) - 10 * np.log10(mse))


def ssim(pred, gt, data_range=1.0):
    return float(np.mean([structural_similarity(pred[i], gt[i], data_range=data_range)
                          for i in range(pred.shape[0])]))


def ncc(pred, gt, eps=1e-12):
    x = np.asarray(pred, np.float64).ravel()
    y = np.asarray(gt, np.float64).ravel()
    x, y = x - x.mean(), y - y.mean()
    return float(np.sum(x * y) / (np.sqrt(np.sum(x * x) * np.sum(y * y)) + eps))


def _hists(pred, gt, bins=128):
    p, edges = np.histogram(np.asarray(pred, np.float64).ravel(), bins=bins, range=RANGE)
    q, _ = np.histogram(np.asarray(gt, np.float64).ravel(), bins=bins, range=RANGE)
    p, q = p.astype(np.float64), q.astype(np.float64)
    return p / p.sum(), q / q.sum(), float(edges[1] - edges[0])


def wasserstein(pred, gt, bins=128):
    p, q, w = _hists(pred, gt, bins)
    return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))) * w)


def nmi(pred, gt, bins=64, eps=1e-12):
    joint, _, _ = np.histogram2d(np.asarray(pred, np.float64).ravel(), np.asarray(gt, np.float64).ravel(),
                                 bins=bins, range=[RANGE, RANGE])
    joint = joint / joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    mi = np.sum(joint[nz] * np.log((joint[nz] + eps) / ((px @ py)[nz] + eps)))
    hx, hy = -np.sum(px * np.log(px + eps)), -np.sum(py * np.log(py + eps))
    return float(mi / (np.sqrt(hx * hy) + eps))


def hist_ncc(pred, gt, bins=128):
    p, q, _ = _hists(pred, gt, bins)
    return ncc(p, q)


def bhattacharyya(pred, gt, bins=128, eps=1e-12):
    p, q, _ = _hists(pred, gt, bins)
    return float(-np.log(np.sum(np.sqrt(p * q)) + eps))


def js_divergence(pred, gt, bins=128, eps=1e-12):
    p, q, _ = _hists(pred, gt, bins)
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log((p + eps) / (m + eps))) + np.sum(q * np.log((q + eps) / (m + eps)))))


METRICS = {"psnr": psnr, "ssim": ssim, "ncc": ncc, "wasserstein": wasserstein, "nmi": nmi,
           "hist_ncc": hist_ncc, "bhattacharyya": bhattacharyya, "js_divergence": js_divergence}


def all_metrics(pred, gt):
    pred, gt = np.asarray(pred, np.float32), np.asarray(gt, np.float32)
    return {k: f(pred, gt) for k, f in METRICS.items()}
