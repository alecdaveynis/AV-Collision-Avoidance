"""Train and evaluate the learned TTC network.

    python src/main.py            # 1) produce output/<drive>/tracks.csv for every drive
    python src/train_ttc.py       # 2) cross-validate, plot, save models/ttc_gru.pt
    python src/main.py --ttc fused   # 3) run the pipeline with the learned TTC

Evaluation (all numbers are out-of-fold -- the model never sees the frames it is
scored on):
  * blocked time cross-validation on all drives pooled: each drive's timeline is
    cut into K contiguous blocks; training samples within (T + W) frames of the
    held-out block are purged so no input/label window overlaps a test window.
  * leave-one-drive-out, reported separately (low-powered if a drive has few
    closing events -- the sample counts are printed).
Compared methods:
  learned        tau prior + GRU residual, camera + ego speed (no LiDAR at inference)
  tau prior      the physics estimate alone: 1 / slope of log box height (camera only)
  analytic LiDAR range-rate over the previous W frames
  analytic mono  monocular range-rate over the previous W frames (camera-only reference)
  constant       always TTC_MAX
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import ttc_model
from ttc_data import Samples, build_samples

THRESHOLD = config.TTC_SLOW      # "needs braking attention" if TTC < 3 s


# ------------------------------------------------------------------ training ----
def fit(train, epochs=25, seed=0, lr=3e-3, weight_decay=1e-3, batch=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = ttc_model.TTCNet()
    flat = train.X.reshape(-1, ttc_model.FEATURE_DIM)
    net.mean.copy_(torch.tensor(flat.mean(0)))
    net.std.copy_(torch.tensor(flat.std(0) + 1e-6))

    X = torch.tensor(train.X)
    y = torch.log(torch.tensor(train.ttc, dtype=torch.float32))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss(delta=0.3)
    n = len(X)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = X[idx]
            # light augmentation: jitter in normalised space
            xb = xb + torch.randn_like(xb) * net.std * 0.03
            loss = loss_fn(net(xb), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    return net


def predict(net, samples):
    if len(samples) == 0:
        return np.zeros(0)
    return net.predict_ttc(samples.X)


# ------------------------------------------------------------------- metrics ----
def prf(label, pred, thr=THRESHOLD):
    y, p = label < thr, pred < thr
    tp, fp, fn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum()
    prec = tp / (tp + fp) if tp + fp else float('nan')
    rec = tp / (tp + fn) if tp + fn else float('nan')
    f1 = 2 * prec * rec / (prec + rec) if tp else (0.0 if (tp + fp + fn) else float('nan'))
    return prec, rec, f1, int(y.sum()), int(fp)


def score(label, pred):
    """Dictionary of metrics for one method."""
    log_err = np.abs(np.log(pred) - np.log(label))
    closing = label < 6.0
    prec, rec, f1, n_pos, fp = prf(label, pred)
    return {
        'log_mae': float(log_err.mean()),
        'mae_closing_s': float(np.abs(pred - label)[closing].mean()) if closing.any() else float('nan'),
        'precision': prec, 'recall': rec, 'f1': f1, 'n_pos': n_pos, 'false_pos': fp,
    }


def table(label, preds, title):
    lines = [title, f"  n={len(label)} samples, {int((label < THRESHOLD).sum())} with TTC<{THRESHOLD:.0f}s, "
                    f"{int((label < 6).sum())} with TTC<6s",
             f"  {'method':<22}{'logMAE':>8}{'MAE<6s':>9}{'prec':>7}{'recall':>8}{'F1':>7}{'FP':>5}"]
    results = {}
    for name, p in preds.items():
        m = score(label, np.clip(p, config.TTC_MIN, config.TTC_MAX))
        results[name] = m
        lines.append(f"  {name:<22}{m['log_mae']:>8.3f}{m['mae_closing_s']:>8.2f}s"
                     f"{m['precision']:>7.2f}{m['recall']:>8.2f}{m['f1']:>7.2f}{m['false_pos']:>5d}")
    return lines, results


# ---------------------------------------------------------------- CV schemes ----
def blocked_cv(samples, k, epochs):
    """Out-of-fold predictions with time-blocked folds and purging."""
    T, W = ttc_model.SEQ_LEN, config.TTC_LABEL_WINDOW
    margin = T + W
    drive = np.array(samples.drive)
    frame = np.array(samples.frame)
    block = np.zeros(len(samples), dtype=int)
    for d in np.unique(drive):
        sel = drive == d
        lo, hi = frame[sel].min(), frame[sel].max() + 1
        edges = np.linspace(lo, hi, k + 1)
        block[sel] = np.clip(np.searchsorted(edges, frame[sel], side='right') - 1, 0, k - 1)

    oof = np.full(len(samples), np.nan)
    for b in range(k):
        test = block == b
        train = np.zeros(len(samples), dtype=bool)
        for d in np.unique(drive):
            sel = drive == d
            fb = frame[sel & test]
            if len(fb) == 0:
                train |= sel & ~test
                continue
            a, z = fb.min(), fb.max()
            train |= sel & ~test & ((frame < a - margin) | (frame > z + margin))
        if train.sum() < 30 or test.sum() == 0:
            continue
        net = fit(samples.subset(train), epochs=epochs)
        oof[test] = predict(net, samples.subset(test))
    return oof


def leave_one_drive_out(samples, epochs):
    oof = np.full(len(samples), np.nan)
    drive = np.array(samples.drive)
    for d in np.unique(drive):
        test = drive == d
        if (~test).sum() < 30:
            continue
        net = fit(samples.subset(~test), epochs=epochs)
        oof[test] = predict(net, samples.subset(test))
    return oof


# --------------------------------------------------------------------- plots ----
def plot_results(label, preds, results, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    lim = (config.TTC_MIN * 0.8, config.TTC_MAX * 1.2)
    for a, name in zip(ax[:2], ('learned', 'analytic LiDAR (past)')):
        a.scatter(label, np.clip(preds[name], config.TTC_MIN, config.TTC_MAX), s=10, alpha=0.35)
        a.plot(lim, lim, 'r--', lw=1)
        a.axvline(THRESHOLD, color='gray', ls=':'); a.axhline(THRESHOLD, color='gray', ls=':')
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(lim); a.set_ylim(lim)
        a.set_xlabel('hindsight TTC label (s)'); a.set_ylabel('predicted TTC (s)')
        a.set_title(f'{name}')
    names = list(results)
    f1 = [results[n]['f1'] for n in names]
    ax[2].bar(range(len(names)), np.nan_to_num(f1), color=['tab:green', 'tab:olive', 'tab:blue', 'tab:orange', 'gray'][:len(names)])
    ax[2].set_xticks(range(len(names))); ax[2].set_xticklabels(names, rotation=20, ha='right')
    ax[2].set_ylabel(f'F1 for "TTC < {THRESHOLD:.0f} s"'); ax[2].set_ylim(0, 1)
    ax[2].set_title('Out-of-fold detection of closing objects')
    fig.suptitle('Learned TTC (out-of-fold, blocked time cross-validation)')
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ---------------------------------------------------------------------- main ----
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--runs-dir', type=Path, default=config.OUTPUT_DIR)
    p.add_argument('--out', type=Path, default=config.TTC_MODEL_PATH)
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--folds', type=int, default=4)
    p.add_argument('--plot', type=Path, default=config.ROOT / 'docs' / 'ttc_eval.png')
    args = p.parse_args()

    csvs = sorted(args.runs_dir.glob('*/tracks.csv'))
    if not csvs:
        sys.exit(f"No tracks.csv under {args.runs_dir}. Run `python src/main.py` first.")
    per_drive = [build_samples(c) for c in csvs]
    samples = Samples.concat(per_drive)
    out = [f"Samples: {len(samples)} total"]
    for c, s in zip(csvs, per_drive):
        out.append(f"  {c.parent.name}: {len(s)} samples, {int((s.ttc < THRESHOLD).sum())} with TTC<{THRESHOLD:.0f}s")
    if len(samples) < 60:
        sys.exit("Too few samples to train. Process more drives.")

    # --- blocked time CV (primary) --------------------------------------------------
    oof = blocked_cv(samples, args.folds, args.epochs)
    ok = np.isfinite(oof)
    s = samples.subset(ok)
    preds = {'learned': oof[ok], 'tau prior (camera)': ttc_model.tau_ttc(s.X),
             'analytic LiDAR (past)': s.base_lidar, 'analytic mono (past)': s.base_mono,
             'constant (10 s)': np.full(len(s), config.TTC_MAX)}
    lines, results = table(s.ttc, preds, f"\n== Blocked time CV, {args.folds} folds, all drives pooled ==")
    out += lines

    drive = np.array(s.drive)
    for d in np.unique(drive):
        sel = drive == d
        l2, _ = table(s.ttc[sel], {k: v[sel] for k, v in preds.items()}, f"\n-- {d} only (out-of-fold) --")
        out += l2

    # --- leave-one-drive-out (secondary) ----------------------------------------------
    if len(per_drive) >= 2:
        oof2 = leave_one_drive_out(samples, args.epochs)
        ok2 = np.isfinite(oof2)
        s2 = samples.subset(ok2)
        l3, _ = table(s2.ttc, {'learned': oof2[ok2], 'tau prior (camera)': ttc_model.tau_ttc(s2.X),
                               'analytic LiDAR (past)': s2.base_lidar,
                               'analytic mono (past)': s2.base_mono},
                      "\n== Leave-one-drive-out (train on the other drives) ==")
        out += l3
        out.append("  (low-powered when the held-out drive has few TTC<3s events)")

    # --- final model on everything ------------------------------------------------------
    net = fit(samples, epochs=args.epochs)
    ttc_model.save_model(net, args.out)
    out.append(f"\nFinal model trained on all {len(samples)} samples -> {args.out}")
    out.append("Note: running the pipeline on these same drives with that model is in-sample.")

    text = '\n'.join(out)
    print(text)
    (args.runs_dir / 'ttc_summary.txt').write_text(text + '\n')
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    plot_results(s.ttc, preds, results, args.plot)
    print(f"Plot: {args.plot}")


if __name__ == '__main__':
    main()
