"""Train JointNet on synthetic stick figures (PyTorch; runs on a Kaggle GPU or, slowly, on a laptop CPU).

    python -m ml.train_joints --out out/models/run1                       # full run (GPU)
    python -m ml.train_joints --smoke --out out/models/smoke              # a minute on a CPU: checks the pipeline

Steps: generate the drawings with sandbox.synth (in parallel, cached as npz shards), train with augmentation
(mirror with l_/r_ swapped, small rotation / scale / shift, ink strength, blur, thicker strokes, noise),
keep the best weights by synthetic validation error, export them for numpy (sandbox/joint_net.py) and score
the real drawings in test_real.npz if it is there.

Loss: cross-entropy between each joint's softmax heatmap and a Gaussian at its label. Each arm pair and each
leg pair is scored both ways round and the better one counts (synth.SIDE_PAIRS): one drawing doesn't say
which arm is the left one, and the geometry fit decides that from the 3D pose.
Errors are reported as % of the figure's size (the larger side of its joints' bounding box), sides aligned.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.model import STRIDE, JointNet, export_numpy, to_ink  # noqa: E402
from sandbox.joint_net import JointNet as NumpyJointNet, align_sides  # noqa: E402
from sandbox.synth import MIRROR, SIDE_PAIRS  # noqa: E402

PCK_LEVELS = (0.03, 0.05, 0.10)


# ---------------------------------------------------------------- data

def _make_shard(job):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    path, n, seed, size = job
    path = pathlib.Path(path)
    if not path.exists():
        from sandbox.synth import generate
        imgs, joints, radii, _ = generate(n, seed=seed, size=size)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, images=imgs, joints=joints)
        tmp.replace(path)
    return str(path)


def synthetic(n, seed, size, cache, workers, shard=2000, tag="train"):
    """n drawings (images uint8 (n, S, S), joints float32 (n, 11, 2)); shards cached in `cache`."""
    cache = pathlib.Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [(str(cache / f"{tag}_s{seed}_{size}_{k:03d}.npz"), min(shard, n - k * shard), seed * 1000 + k, size)
            for k in range(math.ceil(n / shard))]
    t = time.time()
    if workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(_make_shard, jobs)):
                print(f"  {tag}: shard {i + 1}/{len(jobs)}  {time.time() - t:.0f}s", flush=True)
    else:
        for i, j in enumerate(jobs):
            _make_shard(j)
            print(f"  {tag}: shard {i + 1}/{len(jobs)}  {time.time() - t:.0f}s", flush=True)
    imgs, joints = [], []
    for p, *_ in jobs:
        with np.load(p) as z:
            imgs.append(z["images"])
            joints.append(z["joints"])
    return np.concatenate(imgs), np.concatenate(joints)


# ---------------------------------------------------------------- augmentation (on the device, per batch)

def augment(ink, J, gen):
    """ink (B, 1, S, S), J (B, 11, 2) px. Returns augmented copies, labels moved to match."""
    B, _, S, _ = ink.shape
    dev = ink.device
    rnd = lambda *shape: torch.rand(*shape, generator=gen, device=dev)  # noqa: E731
    # mirror half, swapping the sides
    m = rnd(B) < 0.5
    ink = torch.where(m[:, None, None, None], ink.flip(-1), ink)
    Jm = J[:, MIRROR].clone()
    Jm[..., 0] = S - Jm[..., 0]
    J = torch.where(m[:, None, None], Jm, J)
    # small rotation / scale / shift. affine_grid: input point u_in = A u_out + t (normalised, align_corners=False)
    ang = (rnd(B) * 2 - 1) * math.radians(12)
    sc = torch.exp((rnd(B) * 2 - 1) * math.log(1.12))
    tr = (rnd(B, 2) * 2 - 1) * 0.06
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    A = torch.stack([torch.stack([cos, -sin], 1), torch.stack([sin, cos], 1)], 1)          # (B, 2, 2)
    theta = torch.cat([A, tr[:, :, None]], 2)
    grid = F.affine_grid(theta, ink.shape, align_corners=False)
    ink = F.grid_sample(ink, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    u = J / S * 2 - 1
    u = torch.linalg.solve(A, (u - tr[:, None, :]).transpose(1, 2)).transpose(1, 2)        # u_out = A^-1 (u_in - t)
    J = (u + 1) / 2 * S
    # thicker strokes / blur / ink strength / noise
    thick = rnd(B) < 0.2
    ink = torch.where(thick[:, None, None, None], F.max_pool2d(ink, 3, 1, 1), ink)
    blur = rnd(B) < 0.25
    ink = torch.where(blur[:, None, None, None], F.avg_pool2d(ink, 3, 1, 1), ink)
    ink = ink * (0.55 + 0.5 * rnd(B, 1, 1, 1))
    noisy = (rnd(B, 1, 1, 1) < 0.3).float()
    ink = ink + noisy * 0.04 * torch.randn(ink.shape, generator=gen, device=dev)
    return ink.clamp(0, 1), J


# ---------------------------------------------------------------- loss and decoding

def targets(J, h, w, sigma_px):
    """Gaussian target distributions (B, 11, h, w) for joints J (B, 11, 2) px."""
    dev = J.device
    xs = (torch.arange(w, device=dev) + 0.5) * STRIDE
    ys = (torch.arange(h, device=dev) + 0.5) * STRIDE
    dx = xs[None, None, None, :] - J[..., 0, None, None]
    dy = ys[None, None, :, None] - J[..., 1, None, None]
    g = torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma_px ** 2))
    return g / g.sum((-2, -1), keepdim=True).clamp_min(1e-12)


def loss_fn(logits, J, sigma_px):
    B, K, h, w = logits.shape
    logp = F.log_softmax(logits.float().reshape(B, K, -1), -1)
    t = targets(J, h, w, sigma_px).reshape(B, K, -1)
    ce = lambda pi, ti: -(t[:, ti] * logp[:, pi]).sum(-1).sum(-1)  # noqa: E731   (B,)
    paired = sorted(i for a, b in SIDE_PAIRS for i in a + b)
    single = [i for i in range(K) if i not in paired]
    total = ce(single, single)
    for a, b in SIDE_PAIRS:
        total = total + torch.minimum(ce(a + b, a + b), ce(a + b, b + a))
    return total.mean() / K


@torch.no_grad()
def decode_torch(logits, radius=2):
    """Same as sandbox.joint_net.decode, batched: (B, J, h, w) -> (B, J, 2) px."""
    B, K, h, w = logits.shape
    p = F.softmax(logits.float().reshape(B, K, -1), -1).reshape(B, K, h, w)
    idx = p.reshape(B, K, -1).argmax(-1)
    r0, c0 = (idx // w).float(), (idx % w).float()
    rr = torch.arange(h, device=p.device).float()[None, None, :, None]
    cc = torch.arange(w, device=p.device).float()[None, None, None, :]
    win = ((rr - r0[..., None, None]).abs() <= radius) & ((cc - c0[..., None, None]).abs() <= radius)
    q = p * win
    m = q.sum((-2, -1)).clamp_min(1e-12)
    x = (q * (cc + 0.5)).sum((-2, -1)) / m * STRIDE
    y = (q * (rr + 0.5)).sum((-2, -1)) / m * STRIDE
    return torch.stack([x, y], -1)


def score(pred, true, valid=None):
    """Errors as a fraction of figure size, sides aligned per pair. Returns a metrics dict."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    valid = np.ones(true.shape[:2], bool) if valid is None else np.asarray(valid, bool)
    errs = np.full(true.shape[:2], np.nan)
    for i in range(len(true)):
        v = valid[i]
        ext = true[i][v].max(0) - true[i][v].min(0)
        P = align_sides(np.where(v[:, None], true[i], pred[i]), pred[i])
        errs[i, v] = np.linalg.norm(P[v] - true[i][v], axis=1) / max(float(ext.max()), 1e-6)
    e = errs[~np.isnan(errs)]
    out = {"mean_pct": 100 * float(e.mean()), "median_pct": 100 * float(np.median(e)),
           "per_drawing_pct": [100 * float(np.nanmean(r)) for r in errs],
           "per_joint_pct": [100 * float(np.nanmean(c)) if np.any(~np.isnan(c)) else None for c in errs.T]}
    for lv in PCK_LEVELS:
        out[f"pck_{int(lv * 100)}"] = float((e <= lv).mean())
    return out


@torch.no_grad()
def predict_all(model, imgs, dev, batch=256):
    model.eval()
    out = []
    for k in range(0, len(imgs), batch):
        x = to_ink(torch.as_tensor(imgs[k:k + batch]).to(dev))
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            out.append(decode_torch(model(x)).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "out" / "models" / "run"))
    ap.add_argument("--cache", default=str(ROOT / "out" / "training" / "cache"))
    ap.add_argument("--real", default=str(ROOT / "out" / "training" / "test_real.npz"))
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--n-val", type=int, default=2_000)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--sigma", type=float, default=5.0, help="target Gaussian, px")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-minutes", type=float, default=600, help="stop training after this long")
    ap.add_argument("--smoke", action="store_true", help="tiny run that checks the pipeline end to end")
    a = ap.parse_args(argv)
    if a.smoke:
        a.n_train, a.n_val, a.epochs, a.batch, a.workers = 256, 64, 2, 32, 1
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    print(f"device {dev}{' ' + torch.cuda.get_device_name(0) if dev.type == 'cuda' else ''}; "
          f"{a.n_train} train / {a.n_val} val drawings at {a.size} px", flush=True)

    t0 = time.time()
    Xtr, Jtr = synthetic(a.n_train, a.seed, a.size, a.cache, a.workers, tag="train")
    Xva, Jva = synthetic(a.n_val, a.seed + 500, a.size, a.cache, a.workers, tag="val")
    print(f"data ready in {time.time() - t0:.0f}s", flush=True)
    Xtr_d = torch.as_tensor(Xtr).to(dev)               # uint8 on the GPU: ~3.7 GB for 100k at 192 px
    Jtr_d = torch.as_tensor(Jtr).to(dev)

    model = JointNet().to(dev)
    steps_per_epoch = math.ceil(len(Xtr) / a.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    history, best = [], None
    t_train = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen, device=dev)
        tot, t = 0.0, time.time()
        for k in range(steps_per_epoch):
            idx = perm[k * a.batch:(k + 1) * a.batch]
            x, J = augment(to_ink(Xtr_d[idx]), Jtr_d[idx], gen)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                logits = model(x)
            loss = loss_fn(logits, J, a.sigma)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach())
        val = score(predict_all(model, Xva, dev), Jva)
        row = {"epoch": ep + 1, "loss": tot / steps_per_epoch, "val_mean_pct": val["mean_pct"],
               "val_pck_5": val["pck_5"], "seconds": time.time() - t}
        history.append(row)
        print(json.dumps(row), flush=True)
        if best is None or val["mean_pct"] < best:
            best = val["mean_pct"]
            torch.save(model.state_dict(), out / "joint_net.pt")
        if (time.time() - t_train) / 60 > a.max_minutes:
            print("time limit reached, stopping", flush=True)
            break

    model.load_state_dict(torch.load(out / "joint_net.pt", map_location=dev))
    export_numpy(model, out / "joint_net.npz", extra={"size": a.size, "sigma": a.sigma})
    model.to(dev)
    report = {"args": vars(a), "device": str(dev), "history": history,
              "val": score(predict_all(model, Xva, dev), Jva), "train_minutes": (time.time() - t_train) / 60}
    real = pathlib.Path(a.real)
    if real.is_file():
        with np.load(real) as z:
            imgs, joints, valid, names = z["images"], z["joints"], z["valid"], [str(n) for n in z["names"]]
        net = NumpyJointNet.load(out / "joint_net.npz")      # the exact path the sandbox will use
        P1 = np.array([net.predict(im, mirror_too=False)[0] for im in imgs])
        P2 = np.array([net.predict(im, mirror_too=True)[0] for im in imgs])
        report["real"] = {"names": names, "plain": score(P1, joints, valid), "with_mirror": score(P2, joints, valid)}
        np.savez(out / "real_predictions.npz", plain=P1, with_mirror=P2, names=np.array(names))
        try:
            sys.path.insert(0, str(ROOT / "tools"))
            from make_training_set import preview
            preview(out / "real_predictions.png", imgs, P2, cols=min(8, len(imgs)),
                    title="model on the real drawings (with mirror averaging)")
        except Exception as e:                               # a picture is nice to have, not essential
            print(f"preview skipped: {e}")
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("VAL", json.dumps({k: v for k, v in report["val"].items() if not k.startswith("per_")}))
    if "real" in report:
        for k in ("plain", "with_mirror"):
            r = report["real"][k]
            print(f"REAL {k}: mean {r['mean_pct']:.1f}% of figure size, per drawing "
                  + ", ".join(f"{n} {e:.1f}%" for n, e in zip(report["real"]["names"], r["per_drawing_pct"])))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
