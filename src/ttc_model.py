"""Learned time-to-collision.

A small GRU reads the last ``TTC_SEQ_LEN`` frames of one tracked object and
predicts log(TTC). Inputs are *camera + ego-speed only* (box geometry, size
growth rate, class group, monocular range, ego speed) -- no LiDAR. LiDAR is used
only offline, to build training labels from what actually happened next.

Physics-informed: for an object of constant size, TTC = 1 / (d log(box height)/dt)
("tau"). The model starts from that estimate (least-squares slope of log box
height over the window) and learns a bounded residual correction on top, with the
output layer initialised to zero -- so before training it *is* the tau estimate and
training can only improve on it where the data supports it.
"""
from collections import deque

import numpy as np
import torch
import torch.nn as nn

import config

SEQ_LEN = config.TTC_SEQ_LEN
BASE_DIM = 9      # per-frame features
FEATURE_DIM = 12  # + d(log h)/dt, d(log w)/dt, elapsed time in window (s)
MAX_RESIDUAL = 1.5  # |log-TTC correction| bound (a factor of ~4.5)


def class_group(cls):
    """0: person/bicycle (vulnerable), 1: car/motorcycle, 2: bus/truck."""
    return 0 if cls in (0, 1) else (2 if cls in (5, 7) else 1)


def frame_features(box, img_shape, cls, mono_dist, ego_speed):
    """Per-frame feature vector (BASE_DIM,). Pure function of one frame's data."""
    h_img, w_img = img_shape[:2]
    x1, y1, x2, y2 = box
    bh, bw = max(y2 - y1, 1.0), max(x2 - x1, 1.0)
    onehot = np.zeros(3)
    onehot[class_group(cls)] = 1.0
    return np.array([
        np.log(bh),
        np.log(bw),
        (x1 + x2) / 2.0 / w_img - 0.5,
        y2 / h_img,
        np.log(max(mono_dist, 0.5)),
        0.0 if ego_speed is None else ego_speed,
        *onehot,
    ], dtype=np.float32)


def build_sequence(base, dts):
    """(T, BASE_DIM) base features + (T,) dts -> (T, FEATURE_DIM).

    Adds growth rates of log height/width and the elapsed time in the window.
    dts[j] is the time between frame j-1 and j. The first frame reuses the first
    available rate so every row has a defined derivative.
    """
    base = np.asarray(base, dtype=np.float32)
    dts = np.maximum(np.asarray(dts, dtype=np.float32), 1e-3)
    rates = np.diff(base[:, :2], axis=0) / dts[1:, None]
    rates = np.clip(rates, -3.0, 3.0)
    rates = np.vstack([rates[:1], rates])
    elapsed = np.concatenate([[0.0], np.cumsum(dts[1:])]).astype(np.float32)
    return np.hstack([base, rates, elapsed[:, None]]).astype(np.float32)


def tau_log_ttc(x):
    """Physics prior: log TTC from the least-squares slope of log box height.

    x: (B, T, FEATURE_DIM) raw features (tensor). Returns (B,) log TTC; objects
    that are not growing fast enough to close within TTC_MAX get log(TTC_MAX).
    """
    y = x[..., 0]                       # log box height
    t = x[..., 11]                      # elapsed time (s)
    tc = t - t.mean(dim=1, keepdim=True)
    yc = y - y.mean(dim=1, keepdim=True)
    slope = (tc * yc).sum(dim=1) / (tc * tc).sum(dim=1).clamp(min=1e-6)   # = closing / range
    slope = slope.clamp(min=1.0 / config.TTC_MAX)
    return (-torch.log(slope)).clamp(min=float(np.log(config.TTC_MIN)))


def tau_ttc(x):
    """Numpy convenience wrapper: tau-based TTC (s) for (N, T, FEATURE_DIM)."""
    return torch.exp(tau_log_ttc(torch.as_tensor(x, dtype=torch.float32))).numpy()


class TTCNet(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.register_buffer('mean', torch.zeros(FEATURE_DIM))
        self.register_buffer('std', torch.ones(FEATURE_DIM))
        self.gru = nn.GRU(FEATURE_DIM, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.head[-1].weight)    # start exactly at the tau prior
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        """x: (B, T, FEATURE_DIM) raw features -> log TTC (B,)."""
        prior = tau_log_ttc(x)
        _, h = self.gru((x - self.mean) / self.std)
        residual = MAX_RESIDUAL * torch.tanh(self.head(h[-1]).squeeze(-1))
        return prior + residual

    def predict_ttc(self, x):
        with torch.no_grad():
            log_ttc = self.forward(torch.as_tensor(x, dtype=torch.float32))
        return torch.exp(log_ttc).clamp(config.TTC_MIN, config.TTC_MAX).numpy()


def save_model(net, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': net.state_dict(), 'seq_len': SEQ_LEN,
                'feature_dim': FEATURE_DIM}, path)


def load_model(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if ckpt['seq_len'] != SEQ_LEN or ckpt['feature_dim'] != FEATURE_DIM:
        raise ValueError('TTC checkpoint does not match the current feature layout')
    net = TTCNet()
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    return net


class LearnedTTC:
    """Causal per-track inference: keeps the last SEQ_LEN frames of every track."""

    def __init__(self, net):
        self.net = net
        self._hist = {}   # id -> {'f': deque, 'dt': deque, 'last': frame idx}

    @classmethod
    def from_file(cls, path):
        return cls(load_model(path))

    def update(self, tracks, ego_speed, dt, frame_idx, img_shape):
        """Sets t['ttc_learned'] (float or None) on every track dict."""
        batch, owners = [], []
        for t in tracks:
            t['ttc_learned'] = None
            h = self._hist.get(t['id'])
            if h is None or frame_idx - h['last'] != 1:   # new track or gap -> restart
                h = {'f': deque(maxlen=SEQ_LEN), 'dt': deque(maxlen=SEQ_LEN), 'last': frame_idx}
                self._hist[t['id']] = h
            h['f'].append(frame_features(t['box'], img_shape, t['class'], t['mono_dist'], ego_speed))
            h['dt'].append(dt)
            h['last'] = frame_idx
            if len(h['f']) == SEQ_LEN:
                batch.append(build_sequence(list(h['f']), list(h['dt'])))
                owners.append(t)
        if batch:
            preds = self.net.predict_ttc(np.stack(batch))
            for t, p in zip(owners, preds):
                t['ttc_learned'] = float(p)
        for k in [k for k, h in self._hist.items() if frame_idx - h['last'] > config.TRACK_MAX_AGE]:
            del self._hist[k]
