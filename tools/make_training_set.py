"""Generate synthetic stick-figure drawings with exact joint labels (training data for a joint model).

    python tools/make_training_set.py --n 50000 --out out/training --workers 4  # shards of 5000
    python tools/make_training_set.py --n 64 --out out/training --preview-only  # just look at a few

Writes out/training/train_000.npz ... (images uint8 (n, S, S), joints float32 (n, 11, 2) in pixels,
head_r float32 (n,), joint_names) + train_000.jsonl (pose, camera, style per sample), and preview.png.
Pure numpy: it runs the same on this laptop or in a cloud notebook. Each shard has its own seed, so
shards can be made in parallel or resumed.
"""
import argparse
import json
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sandbox.sketch_detect import PLACED  # noqa: E402
from sandbox.synth import PARENT, generate  # noqa: E402


def preview(path, imgs, joints, cols=8, title=""):
    n = len(imgs)
    rows = int(np.ceil(n / cols))
    fig = Figure(figsize=(cols * 1.6, rows * 1.6 + 0.4))
    FigureCanvasAgg(fig)
    for i in range(n):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=255, extent=(0, imgs[i].shape[1], imgs[i].shape[0], 0))
        J = joints[i]
        for k, j in enumerate(PLACED):
            p = PARENT[j]
            if p is not None:
                q = PLACED.index(p)
                c = "#1f77b4" if j.startswith("l_") else "#d62728" if j.startswith("r_") else "#2ca02c"
                ax.plot([J[q, 0], J[k, 0]], [J[q, 1], J[k, 1]], "-", color=c, lw=0.9, alpha=0.8)
        ax.plot(J[:, 0], J[:, 1], "o", ms=1.6, color="#ff7f0e")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title or "synthetic drawings; labels: blue = left, red = right", fontsize=8)
    fig.savefig(path, dpi=90, facecolor="white")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--shard", type=int, default=5000)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "out" / "training"))
    ap.add_argument("--prefix", default="train")
    ap.add_argument("--preview-only", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="shards made in parallel (one per CPU core)")
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.preview_only:
        imgs, joints, _, meta = generate(min(args.n, 64), seed=args.seed, size=args.size)
        preview(out / "preview.png", imgs, joints)
        print(f"wrote {out / 'preview.png'}  families: {sorted({m['family'] for m in meta})}")
        return
    n_shards = int(np.ceil(args.n / args.shard))
    jobs = [(str(out), args.prefix, s, min(args.shard, args.n - s * args.shard), args.seed * 100003 + s, args.size)
            for s in range(n_shards)]
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for line in pool.imap_unordered(make_shard, jobs):
                print(line, flush=True)
    else:
        for job in jobs:
            print(make_shard(job), flush=True)


def make_shard(job) -> str:
    out, prefix, s, n, seed, size = job
    out = pathlib.Path(out)
    path = out / f"{prefix}_{s:03d}.npz"
    if path.exists():
        return f"{path.name} exists, skipped"
    t = time.time()
    imgs, joints, radii, meta = generate(n, seed=seed, size=size)
    tmp = path.with_suffix(".tmp.npz")                     # a half-written shard never looks finished
    np.savez_compressed(tmp, images=imgs, joints=joints, head_r=radii, joint_names=np.array(PLACED))
    with open(out / f"{prefix}_{s:03d}.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    tmp.replace(path)
    if s == 0:
        preview(out / "preview.png", imgs[:64], joints[:64])
    return f"{path.name}: {n} drawings in {time.time() - t:.0f}s"


if __name__ == "__main__":
    main()
