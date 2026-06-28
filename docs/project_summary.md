# Project Summary

## One-line Summary

CARLA 시뮬레이션 데이터로 학습한 자율주행 인지 모델을 실제 도로 데이터와 엣지 디바이스 환경에 적용하기 위한 Sim2Real 자율주행 프로젝트입니다.

## Goal

고비용 자율주행 시스템 도입 문제를 줄이기 위해 저비용 탈부착형 모듈 시스템 가능성을 검증합니다. 핵심 실증 목표는 시뮬레이션 데이터로 학습한 딥러닝 모델을 엣지 디바이스에서 실시간 추론 가능하도록 최적화하고 검증하는 것입니다.

## Problem

- 실제 주행 데이터 수집과 라벨링에는 비용과 시간이 많이 듭니다.
- 시뮬레이션 학습 모델은 실제 환경으로 이동할 때 Domain Gap이 발생합니다.
- 엣지 디바이스는 GPU 서버보다 연산, 메모리, 발열 제약이 큽니다.

## Main Features

- CARLA 전방 카메라 기반 시뮬레이션 데이터 수집
- U-Net 기반 차선/주행 영역 세그멘테이션
- 데이터 증강과 ResNet backbone을 통한 Domain Gap 완화 실험
- AI-Hub 및 자체 실제 도로 데이터 기반 평가
- YOLO 기반 객체 인식 및 센서 융합 주행
- ONNX export, FP16/INT8 변환
- TensorRT engine build 및 엣지 추론 벤치마크

## Technology Stack

- Python 3.10
- CARLA Simulator
- PyTorch
- U-Net, ResNet34, ResNet50
- Ultralytics YOLO
- ONNX, ONNX Runtime
- TensorRT, CUDA, PyCUDA
- Jetson Nano
- Arduino Uno R3

## Data Flow

```text
CARLA simulation
  -> RGB / semantic camera frames
  -> U-Net masks and YOLO labels
  -> augmentation
  -> PyTorch training
  -> AI-Hub / real-data evaluation
  -> ONNX export
  -> ONNX FP16/INT8 or TensorRT FP16/INT8
  -> benchmark reports and rendered inference videos
```

## Current Code Structure

- `src/data/`: data collection, conversion, augmentation, YOLO label remapping
- `src/models/`: Dataset classes, U-Net variants, training, sample inference
- `src/driving/`: CARLA driving, control, lane tracking, sensor fusion, ACC
- `src/optimization/`: ONNX export, ONNX quantization, TensorRT engine build
- `src/benchmarks/`: PyTorch/ONNX/TensorRT benchmark and video rendering
- `src/utils/`: environment checks, debugging helpers, experimental utilities
- `src/config.py`: shared paths, CARLA connection settings, model paths, training defaults

## Core Modules to Preserve

- `src/data/step09_dl_data_collector.py`, `src/data/step09_advanced_collector.py`: CARLA dataset generation
- `src/models/step10_dataset.py`, `src/models/step10_advanced_dataset.py`: PyTorch dataset loaders
- `src/models/step11_model.py`, `src/models/step11_advanced_model.py`, `src/models/step22_smp_model.py`: segmentation model definitions
- `src/models/step12_train.py`, `src/models/step12_advanced_train.py`: training loops and checkpoint saving
- `src/data/step20_data_augmentation.py`: Sim2Real augmentation
- `src/data/step23_aihub_to_dataset.py`: AI-Hub evaluation conversion
- `src/benchmarks/step24_benchmark_models.py`: PyTorch model benchmark
- `src/optimization/step26_export_onnx.py`: PyTorch to ONNX export
- `src/optimization/step27_quantize_onnx.py`, `src/optimization/step27_build_tensorrt.py`: quantization and TensorRT build
- `src/benchmarks/step31_grand_benchmark.py`: integrated ONNX/TensorRT benchmark
- `src/benchmarks/step32_make_inference_video.py`, `src/benchmarks/step33.py`: inference video rendering

## Test Environment

Unless otherwise noted, benchmark values were measured on the local PC.

| Component | Specification |
|---|---|
| Host OS | Windows 11 Pro |
| Test OS | Ubuntu 22.04.5 LTS on WSL2 |
| CPU | AMD Ryzen 9 7900X, 12 cores / 24 threads |
| Memory | 31.1 GiB host RAM, 15 GiB visible in WSL2 |
| GPU | NVIDIA GeForce RTX 4080 SUPER, 16,376 MiB VRAM |
| NVIDIA driver / CUDA | Driver 591.86, CUDA 13.1 |
| Python | `conda` environment `carla`, Python 3.10.20 |

## Tests

### Segmentation Model Benchmark (`.pth`)

| Model | Dataset | Encoder | mIoU | Checkpoint Size | FPS / Latency |
|---|---|---|---:|---:|---|
| Vanilla U-Net | Original | Custom U-Net | 1.34% | 88.96 MiB | To be reviewed |
| Vanilla U-Net | Augmented | Custom U-Net | 13.63% | 88.96 MiB | To be reviewed |
| ResNet U-Net | Augmented | ResNet34 | 20.88% | 279.95 MiB | See quantization benchmark |
| ResNet U-Net | Augmented | ResNet50 | 25.79% | 372.69 MiB | To be reviewed |

### Quantization Format Benchmark

| Runtime | Format | FPS | Latency | mIoU | Artifact Size | Change |
|---|---|---:|---:|---:|---:|---|
| ONNX Runtime | FP32 | 29.7 | 33.67 ms | 20.88% | 81.78 MiB | Baseline |
| ONNX Runtime | FP16 | 15.3 | 65.36 ms | 20.88% | 46.89 MiB | FPS -48.5% vs ONNX FP32 |
| ONNX Runtime | INT8 | 34.9 | 28.65 ms | 20.66% | 23.70 MiB | FPS +17.5%, mIoU -0.22pp vs ONNX FP32 |
| TensorRT | FP16 | 155.2 | 6.44 ms | 20.88% | 46.98 MiB | FPS +422.6% vs ONNX FP32 |
| TensorRT | INT8 | 180.2 | 5.55 ms | 20.88% | 23.86 MiB | FPS +16.1%, size -49.2% vs TensorRT FP16 |

Review note: the local `ResNet34_Aug.onnx` FP32 artifact is 0.31 MiB, which is much smaller than the FP16 and INT8 artifacts. Re-export the FP32 ONNX file before treating the size comparison as final.

### Jetson Nano Benchmark

The Jetson Nano result is limited to one ResNet34 U-Net TensorRT FP16 test from the presentation materials.

| Device | Model | Runtime | Format | FPS |
|---|---|---|---|---:|
| Yahboom Jetson Nano | ResNet34 U-Net | TensorRT | FP16 | 26.4 |

## Cleanup Notes

- Large generated datasets and model artifacts should not be committed to a normal Git repository.
- Common paths and runtime constants are centralized in `src/config.py`.
- CARLA and TensorRT scripts need prepared external runtimes and should be treated separately from simple Python syntax validation.
- Content belonging to other projects was removed from this repository summary.
