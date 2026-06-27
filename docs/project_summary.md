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

## Tests

### Segmentation Model Benchmark

| Model | Dataset | Encoder | Metric | Result |
|---|---|---|---|---:|
| Vanilla U-Net | Original | Custom U-Net | mIoU | TBD |
| Vanilla U-Net | Augmented | Custom U-Net | mIoU | TBD |
| ResNet U-Net | Augmented | ResNet34 | mIoU | 20.88% |
| ResNet U-Net | Augmented | ResNet50 | mIoU | 25.79% |

### Quantization Format Benchmark

| Runtime | Format | Result |
|---|---|---:|
| ONNX Runtime | FP32 | 29.7 FPS |
| ONNX Runtime | FP16 | 15.3 FPS |
| ONNX Runtime | INT8 | 34.9 FPS |
| TensorRT | FP16 | 155.2 FPS |
| TensorRT | INT8 | 180.2 FPS |
| Jetson Nano TensorRT | FP16 | 26.4 FPS |

## Cleanup Notes

- Large generated datasets and model artifacts should not be committed to a normal Git repository.
- Common paths and runtime constants are centralized in `src/config.py`.
- CARLA and TensorRT scripts need prepared external runtimes and should be treated separately from simple Python syntax validation.
- Content belonging to other projects was removed from this repository summary.
