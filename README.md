# Model Optimization Pipeline: Quantization and Pruning (ResNet-18, CIFAR-10)

Inam ul Haq | 22i-1906 | Section B | Edge Computing & Intelligent Systems, Assignment 3

This repository contains a PyTorch pipeline that trains a ResNet-18 on CIFAR-10 and compresses it in four ways:
post-training static quantization (PTQ), quantization-aware training (QAT), iterative unstructured pruning, and
structured channel pruning. All models are benchmarked with the same script and exported for CPU inference.

## Installation

Python 3.9 or newer.

```
pip install -r requirements.txt
```

A GPU is recommended for training (the scripts use it automatically). Quantized models always run on the CPU.
CIFAR-10 is downloaded automatically into `data/` on the first run.

## Run everything

```
python run_all.py --repo https://github.com/<username>/<repo-name>
```

`python run_all.py --quick` runs 1-epoch versions to check that the pipeline works. Do not use its numbers.

## Run step by step

| Step | Command | Output |
|------|---------|--------|
| 1. Baseline | `python train_baseline.py` | `models/fp32_resnet18.pth` |
| 2. PTQ | `python ptq.py` | `models/ptq_int8.pt` |
| 3. QAT | `python qat.py` | `models/qat_int8.pt` |
| 4. Unstructured pruning | `python prune_unstructured.py` | `models/unstructured_pruned.pth` |
| 5. Structured pruning | `python prune_structured.py` | `models/structured_pruned.pth` |
| 6. ONNX export | `python export_onnx.py` | `models/*.onnx` |
| 7. Benchmark | `python benchmark.py` | `results/benchmark_results.csv`, `results/predictions.csv` |
| 8. Report | `python make_report.py --repo <url>` | `22i-1906_B.pdf` |

Steps 2 and 3 need step 1. Steps 4 and 5 need step 1. Step 6 needs steps 1, 4 and 5. Step 7 uses whatever models exist.

## Inference

```
python inference.py --model fp32 --samples 5
python inference.py --model ptq --image my_picture.png
python inference.py --model onnx_structured
```

Available models: `fp32`, `ptq`, `qat`, `unstructured`, `structured`, `onnx_fp32`, `onnx_unstructured`,
`onnx_structured`, `onnx_int8`.

## Files

| File | Purpose |
|------|---------|
| `common.py` | data loading, model definition, training loop, metrics, unified predictor |
| `train_baseline.py` | FP32 training |
| `ptq.py` | fusion, backend, calibration, INT8 conversion |
| `qat.py` | fake-quant insertion, fine-tuning, INT8 conversion |
| `prune_unstructured.py` | iterative global magnitude pruning, masks made permanent |
| `prune_structured.py` | L1 channel pruning with a rebuilt smaller network |
| `export_onnx.py` | ONNX export and ONNX Runtime INT8 quantization |
| `inference.py` | inference for all models |
| `benchmark.py` | accuracy, size and latency of all models, CSV output |
| `make_report.py` | builds the PDF report from the CSV and JSON results |
| `run_all.py` | runs all steps in order |

## Notes

- The INT8 TorchScript files (`ptq_int8.pt`, `qat_int8.pt`) are tied to the quantization backend they were made
  with (`fbgemm` on x86, `qnnpack` on ARM). Run inference on the same type of CPU.
- Structured pruning only removes the inner channels of each residual block, so the skip connections keep valid
  shapes. The saved checkpoint stores the channel widths so the smaller network can be rebuilt.
- Latency is measured with batch size 1, one CPU thread, 20 warm-up runs and 300 timed runs.
