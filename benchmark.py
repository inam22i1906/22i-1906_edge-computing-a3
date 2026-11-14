"""Deliverable 7: evaluate every available model on the CIFAR-10 test set.

Writes results/benchmark_results.csv (accuracy, size, latency) and results/predictions.csv
(per-image predictions of every model). Latency: batch size 1, one CPU thread.
"""
import argparse
import csv
import os
import platform
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--runs", type=int, default=300)
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
if a.quick:
    a.runs = 50


def latency(pred, images, warmup, runs):
    n = len(images)
    for i in range(warmup):                       # warm-up: first calls are slower (lazy init, caches)
        pred(images[i % n:i % n + 1])
    ts = []
    for i in range(runs):
        x = images[i % n:i % n + 1]
        t0 = time.perf_counter_ns()
        pred(x)
        ts.append((time.perf_counter_ns() - t0) / 1e6)
    ts = np.array(ts)
    return {"mean_ms": ts.mean(), "p50_ms": np.percentile(ts, 50), "p95_ms": np.percentile(ts, 95),
            "p99_ms": np.percentile(ts, 99), "throughput_img_s": 1000.0 / ts.mean()}


ds = get_test_set()
loader = DataLoader(ds, 256, shuffle=False)
labels = torch.tensor(ds.targets)
images = torch.stack([ds[i][0] for i in range(100)])
cpu = platform.processor() or platform.machine()
rows, preds = [], {}

for name in PATHS:
    if not os.path.exists(PATHS[name]):
        print(f"skipping {name}: {PATHS[name]} not found")
        continue
    p = Predictor(name, threads=os.cpu_count())              # fast pass for accuracy
    pr = torch.cat([p(x).argmax(1) for x, _ in loader])
    preds[name] = pr
    acc = 100.0 * (pr == labels).float().mean().item()

    p1 = Predictor(name, threads=1)                          # single thread for latency
    row = {"model": name, "description": DISPLAY[name], "accuracy_pct": round(acc, 2),
           "file_mb": round(file_mb(PATHS[name]), 3), "zip_mb": round(zip_mb(PATHS[name]), 3)}
    if p.model is not None:
        row["params"] = sum(q.numel() for q in p.model.parameters())
        row["nonzero_params"] = sum((q != 0).sum().item() for q in p.model.parameters())
        row["weight_sparsity_pct"] = round(weight_sparsity(p.model), 2)
        row["macs"] = count_macs(p.model)
    row.update({k: round(float(v), 3) for k, v in latency(p1, images, a.warmup, a.runs).items()})
    row.update({"threads": 1, "batch_size": 1, "warmup": a.warmup, "runs": a.runs, "cpu": cpu})
    rows.append(row)
    print(f"{DISPLAY[name]:<22} acc {acc:6.2f}%  p50 {row['p50_ms']:7.3f} ms  p99 {row['p99_ms']:7.3f} ms  "
          f"{row['file_mb']:.2f} MB", flush=True)

fields = ["model", "description", "accuracy_pct", "params", "nonzero_params", "weight_sparsity_pct", "macs",
          "file_mb", "zip_mb", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "throughput_img_s",
          "threads", "batch_size", "warmup", "runs", "cpu"]
with open(os.path.join(RESULTS, "benchmark_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, restval="")
    w.writeheader()
    w.writerows(rows)

names = list(preds)
with open(os.path.join(RESULTS, "predictions.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["index", "label"] + [n + "_pred" for n in names])
    for i in range(len(labels)):
        w.writerow([i, int(labels[i])] + [int(preds[n][i]) for n in names])
print("wrote results/benchmark_results.csv and results/predictions.csv")
