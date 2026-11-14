"""Deliverable 2: builds the short technical report (PDF) from the measured results.

Usage:  python make_report.py --repo https://github.com/<username>/<repo-name>
"""
import argparse
import csv
import json
import os
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from common import DISPLAY, PATHS, RESULTS, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="Inam ul Haq")
ap.add_argument("--roll", default="22i-1906")
ap.add_argument("--section", default="B")
ap.add_argument("--repo", default="https://github.com/<username>/<repo-name>")
a = ap.parse_args()

FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)


def jl(name, default=None):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else default


with open(os.path.join(RESULTS, "benchmark_results.csv")) as fh:
    rows = list(csv.DictReader(fh))
R = {r["model"]: r for r in rows}
core = ["fp32", "ptq", "qat", "unstructured", "structured"]
if [m for m in core if m not in R]:
    sys.exit("benchmark_results.csv is missing some models - run the full pipeline first.")


def g(m, k):
    v = R[m].get(k, "")
    return float(v) if v not in ("", None) else None


def fmt(v, nd=2, scale=1.0):
    return "-" if v is None else f"{v / scale:.{nd}f}"


acc = lambda m: g(m, "accuracy_pct")
p50 = lambda m: g(m, "p50_ms")


def rel(base, new):
    r = base / new
    return f"{r:.2f}x faster" if r >= 1 else f"{1 / r:.2f}x slower"


# ---------------------------------------------------------------- figure (2 x 2)
SHORT = {"fp32": "FP32", "ptq": "PTQ", "qat": "QAT", "unstructured": "Unstr.", "structured": "Struct."}
COL = {"fp32": "#1f77b4", "ptq": "#ff7f0e", "qat": "#2ca02c", "unstructured": "#d62728", "structured": "#9467bd"}
names, cols = [SHORT[m] for m in core], [COL[m] for m in core]
x, w = np.arange(len(core)), 0.38
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0))

av = [acc(m) for m in core]
b = ax[0, 0].bar(names, av, color=cols)
ax[0, 0].bar_label(b, fmt="%.2f", fontsize=7)
ax[0, 0].set_ylim(min(av) - 3, max(av) + 1.5)
ax[0, 0].set_title("Test accuracy (%)", fontsize=9)

ax[0, 1].bar(x - w / 2, [p50(m) for m in core], w, label="p50", color="#1f77b4")
ax[0, 1].bar(x + w / 2, [g(m, "p99_ms") for m in core], w, label="p99", color="#ff7f0e")
ax[0, 1].set_xticks(x)
ax[0, 1].set_xticklabels(names)
ax[0, 1].legend(fontsize=7)
ax[0, 1].set_title("Latency, batch 1, 1 thread (ms)", fontsize=9)

ax[1, 0].bar(x - w / 2, [g(m, "file_mb") for m in core], w, label="file", color="#2ca02c")
ax[1, 0].bar(x + w / 2, [g(m, "zip_mb") for m in core], w, label="zipped", color="#9467bd")
ax[1, 0].set_xticks(x)
ax[1, 0].set_xticklabels(names)
ax[1, 0].legend(fontsize=7)
ax[1, 0].set_title("Checkpoint size (MB)", fontsize=9)

it = jl("unstructured_iterations.json", [])
if it:
    sp = [r["sparsity"] for r in it]
    ax[1, 1].plot(sp, [r["acc_before_ft"] for r in it], "o-", color="#d62728", label="after pruning")
    ax[1, 1].plot(sp, [r["acc_after_ft"] for r in it], "s-", color="#2ca02c", label="after fine-tuning")
    ax[1, 1].set_xlabel("weight sparsity (%)", fontsize=8)
    ax[1, 1].legend(fontsize=7)
ax[1, 1].set_title("Iterative unstructured pruning: accuracy (%)", fontsize=9)
for axis in ax.flat:
    axis.tick_params(labelsize=7)
fig.tight_layout()
fig_path = os.path.join(FIG, "summary.png")
fig.savefig(fig_path, dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- text
ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="Times-Roman", fontSize=9.5, leading=12,
                      textColor=colors.black, spaceAfter=4, alignment=4)
h1 = ParagraphStyle("h1", parent=body, fontName="Times-Bold", fontSize=11.5, leading=14, spaceBefore=6,
                    spaceAfter=3, alignment=0)
title = ParagraphStyle("title", parent=body, fontName="Times-Bold", fontSize=15, leading=18, alignment=1)
center = ParagraphStyle("center", parent=body, alignment=1)
story = []


def P(t, s=body):
    story.append(Paragraph(t, s))


pi, qi = jl("ptq_info.json", {}), jl("qat_info.json", {})
si, hist0 = jl("structured_info.json", {}), jl("history_fp32.json", [])
hist_s = jl("history_structured.json", [])
backend = pi.get("backend", "fbgemm")

P("Model Optimization Pipeline: Quantization and Pruning of ResNet-18 on CIFAR-10", title)
P("Edge Computing &amp; Intelligent Systems - Assignment 3", center)
P(f"{a.name} | Roll no. {a.roll} | Section {a.section}", center)
P(f"Code, models and results: {a.repo}", center)

P("1. Overview", h1)
P("In this assignment I built one pipeline that takes a ResNet-18 trained on CIFAR-10 and produces four "
  "compressed variants: a post-training quantized (PTQ) INT8 model, a quantization-aware trained (QAT) INT8 "
  "model, an unstructured-pruned model and a structured-pruned model. All models share one inference script "
  "and one benchmarking script, and the pruned models and one INT8 model are also exported to ONNX.")

P("2. Implementation", h1)
P(f"<b>Baseline.</b> I used ResNet-18 adapted to 32x32 inputs (3x3 first convolution, no max-pool) and trained it "
  f"for {len(hist0)} epochs with SGD (momentum 0.9, cosine learning rate, random crop and horizontal flip). "
  f"The random seed is 42 everywhere. Environment: Python {platform.python_version()}, PyTorch "
  f"{torch.__version__}, CPU {R['fp32'].get('cpu', '')}.")
P(f"<b>PTQ.</b> I fused Conv+BatchNorm+ReLU, selected the {backend} backend, passed {pi.get('calib_images', 1000)} "
  f"random training images through the observers for calibration, and converted the model to INT8 [1]. "
  f"The model is saved as a TorchScript file so it can run without the training code.")
qat_extra = (" BatchNorm statistics were frozen after 40% of the epochs and the observers after 60%."
             if qi.get("epochs", 0) > 2 else "")
P(f"<b>QAT.</b> I started from the trained FP32 weights, fused the layers, inserted fake-quantization modules "
  f"and fine-tuned for {qi.get('epochs', '-')} epochs (learning rate {qi.get('lr', '-')}).{qat_extra} "
  f"After training I converted the model to a real INT8 model.")
final_target = it[-1]["target"] if it else 0
P(f"<b>Unstructured pruning.</b> I applied global L1 magnitude pruning to all convolution and linear weights in "
  f"{len(it)} iterations, using a cubic sparsity schedule up to {final_target:.0f}% [2][4]. After each iteration "
  f"I fine-tuned the network while the masks were active, and at the end I removed the masks so the zeros are "
  f"permanent in the saved weights.")
P(f"<b>Structured pruning.</b> I ranked the filters of the first convolution of every residual block by L1 norm "
  f"[3], removed {si.get('ratio', 0):.0%} of them, and rebuilt each block with a smaller conv1, BatchNorm and "
  f"conv2. I left the block outputs unchanged so the skip connections [5] still match. The new network has no "
  f"zero-valued channels, and I fine-tuned it for {len(hist_s)} epochs.")
P(f"<b>Benchmark and ONNX.</b> Every model is evaluated on the 10,000 CIFAR-10 test images. Latency uses batch "
  f"size 1, one CPU thread, {R['fp32'].get('warmup', 20)} warm-up runs and {R['fp32'].get('runs', 300)} timed "
  f"runs. The ONNX models run with ONNX Runtime's CPU provider; the INT8 ONNX model comes from ONNX Runtime "
  f"static quantization of the FP32 ONNX file.")

P("3. Results", h1)
hdr = ["Model", "Acc (%)", "Params (M)", "MACs (M)", "Size (MB)", "Zip (MB)", "p50 (ms)", "p99 (ms)", "img/s"]
data = [hdr] + [[DISPLAY[r["model"]], fmt(g(r["model"], "accuracy_pct")), fmt(g(r["model"], "params"), 2, 1e6),
                 fmt(g(r["model"], "macs"), 0, 1e6), fmt(g(r["model"], "file_mb")), fmt(g(r["model"], "zip_mb")),
                 fmt(g(r["model"], "p50_ms"), 3), fmt(g(r["model"], "p99_ms"), 3),
                 fmt(g(r["model"], "throughput_img_s"), 1)] for r in rows]
tbl = Table(data, colWidths=[1.5 * inch] + [0.62 * inch] * 8, repeatRows=1)
tbl.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, 0), "Times-Bold"), ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                         ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
                         ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                         ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                         ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
story += [tbl, Spacer(1, 4)]
P("Table 1: test accuracy, size and single-thread batch-1 latency of every model. Params and MACs are shown "
  "for FP32-format models only; the parameter count of the unstructured model is unchanged because the zeros "
  "are stored.", ParagraphStyle("cap", parent=body, fontSize=8, leading=10))
story.append(Image(fig_path, width=6.6 * inch, height=6.6 * inch * 5.0 / 7.2))

P("4. Findings", h1)
d_ptq, d_qat = acc("ptq") - acc("fp32"), acc("qat") - acc("fp32")
P(f"The FP32 baseline reached {acc('fp32'):.2f}% test accuracy. PTQ gave {acc('ptq'):.2f}% ({d_ptq:+.2f} points) "
  f"and QAT gave {acc('qat'):.2f}% ({d_qat:+.2f} points versus FP32, {acc('qat') - acc('ptq'):+.2f} versus PTQ). "
  f"Both INT8 files are about {g('fp32', 'file_mb') / g('ptq', 'file_mb'):.1f}x smaller than the FP32 checkpoint "
  f"({g('ptq', 'file_mb'):.1f} MB against {g('fp32', 'file_mb'):.1f} MB). The median latency was "
  f"{p50('ptq'):.2f} ms for PTQ and {p50('qat'):.2f} ms for QAT against {p50('fp32'):.2f} ms for FP32, so PTQ "
  f"was {rel(p50('fp32'), p50('ptq'))} and QAT was {rel(p50('fp32'), p50('qat'))} than FP32.")
P(f"Global magnitude pruning reached {g('unstructured', 'weight_sparsity_pct'):.1f}% weight sparsity with "
  f"{acc('unstructured'):.2f}% accuracy ({acc('unstructured') - acc('fp32'):+.2f} points). PyTorch's dense "
  f"kernels do not skip zero weights, so I did not expect a speed-up from this model; the measured median was "
  f"{p50('unstructured'):.2f} ms against {p50('fp32'):.2f} ms for FP32. The saving appears only after "
  f"compression: the zipped checkpoint is {g('unstructured', 'zip_mb'):.1f} MB against "
  f"{g('fp32', 'zip_mb'):.1f} MB.")
P(f"Structured pruning cut the parameters from {si['params_before'] / 1e6:.2f}M to "
  f"{si['params_after'] / 1e6:.2f}M and the MACs from {si['macs_before'] / 1e6:.0f}M to "
  f"{si['macs_after'] / 1e6:.0f}M. Accuracy was {si['acc_after_prune']:.2f}% right after pruning and "
  f"{si['acc_after_ft']:.2f}% after fine-tuning ({si['acc_after_ft'] - si['acc_before']:+.2f} points versus "
  f"FP32). Because the network itself is smaller, its median latency was {p50('structured'):.2f} ms, "
  f"{rel(p50('fp32'), p50('structured'))} than FP32.")
ok = [m for m in core[1:] if acc(m) >= acc("fp32") - 1.0]
if ok:
    best = min(ok, key=p50)
    P(f"Among the compressed models that stay within one accuracy point of FP32, the fastest on this machine was "
      f"{DISPLAY[best]} ({acc(best):.2f}%, {p50(best):.2f} ms median).")
else:
    P("None of the compressed models stayed within one accuracy point of FP32 in this run.")

P("5. Limitations", h1)
P("These numbers come from a single run with one seed on one machine, so small differences (a few tenths of a "
  "point or a few percent of latency) should not be over-interpreted. INT8 speed depends on the CPU and on the "
  f"backend ({backend}); results on ARM or older CPUs can differ. The ONNX INT8 model is produced by ONNX "
  "Runtime from the FP32 model, so it is not the same network as the PTQ and QAT models.")

P("6. Reproduction", h1)
P("Install the packages in requirements.txt and run <font face='Courier'>python run_all.py</font>. It trains, "
  "quantizes, prunes, exports and benchmarks everything, then rebuilds this report. Details are in README.md.")

P("References", h1)
for r in ["[1] B. Jacob et al., Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only "
          "Inference, CVPR 2018.",
          "[2] S. Han et al., Learning both Weights and Connections for Efficient Neural Networks, NeurIPS 2015.",
          "[3] H. Li et al., Pruning Filters for Efficient ConvNets, ICLR 2017.",
          "[4] M. Zhu and S. Gupta, To prune, or not to prune: exploring the efficacy of pruning for model "
          "compression, 2017.",
          "[5] K. He et al., Deep Residual Learning for Image Recognition, CVPR 2016."]:
    P(r, ParagraphStyle("ref", parent=body, fontSize=8.5, leading=10.5, alignment=0, spaceAfter=1))

out = os.path.join(ROOT, f"{a.roll}_{a.section}.pdf")
SimpleDocTemplate(out, pagesize=A4, leftMargin=0.65 * inch, rightMargin=0.65 * inch, topMargin=0.6 * inch,
                  bottomMargin=0.6 * inch, title="Assignment 3 - Model Optimization Pipeline",
                  author=a.name).build(story)
print("report written to", out)
