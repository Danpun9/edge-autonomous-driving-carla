# CARLA 기반 엣지 자율주행 프로젝트

[English README](README.md)

CARLA 시뮬레이션 데이터로 자율주행 인지 모델을 학습하고, 실제 도로 데이터와 엣지 추론 환경에서의 적용 가능성을 실험한 Sim2Real 프로젝트입니다.

이 프로젝트는 U-Net 기반 차선/주행 영역 세그멘테이션, YOLO 기반 객체 인식, CARLA 주행 실험, ONNX/TensorRT 최적화 및 벤치마크 스크립트를 포함합니다.

## 주요 기능

- CARLA RGB 및 semantic camera 데이터 수집
- IPM, 마스크, polynomial fitting 기반 전통적 차선 인식
- 이진/다중 클래스 U-Net 학습 파이프라인
- ResNet34/ResNet50 기반 세그멘테이션 모델 실험
- YOLO 기반 차량, 보행자, 신호등 인식 실험
- U-Net + YOLO 센서 융합 주행, ACC, 수동 override 실험
- 실제 도로 데이터 및 AI-Hub 데이터 변환 스크립트
- PyTorch 모델 ONNX export
- ONNX FP16/INT8 및 TensorRT FP16/INT8 벤치마크
- 추론 결과 영상 렌더링

## 프로젝트 구조

```text
.
├── src/
│   ├── config.py          # 공통 경로, CARLA 접속값, 모델 경로
│   ├── data/              # 데이터 수집, 변환, 증강
│   ├── models/            # Dataset, 모델 정의, 학습
│   ├── driving/           # CARLA 주행, 차선 추적, 센서 융합
│   ├── optimization/      # ONNX export, quantization, TensorRT build
│   ├── benchmarks/        # 모델 벤치마크 및 영상 렌더링
│   └── utils/             # 환경 점검 및 디버그 도구
├── docs/
│   └── project_summary.md
├── carla_yolo.yaml
├── carla_aug_yolo.yaml
├── LICENSE
├── requirements.txt
└── README.md
```

## 실행 환경

기본 실험 환경은 Python 3.10, CARLA, PyTorch, OpenCV, Ultralytics YOLO, ONNX Runtime, TensorRT, PyCUDA, NVIDIA GPU 도구를 사용합니다.

```bash
conda activate carla
python --version
```

PowerShell에서 conda 환경이 정상적으로 활성화되지 않는 경우 다음 방식으로 확인할 수 있습니다.

```bash
conda run -n carla python --version
```

## 설정

공통 설정은 `src/config.py`에서 관리합니다.

개인 환경별 경로나 포트만 바꾸고 싶다면 `src/local_config.py`를 만들고 필요한 상수만 재정의하면 됩니다. 이 파일은 Git에 포함되지 않습니다.

## 실행 예시

아래 명령은 프로젝트 루트에서 실행하는 것을 기준으로 합니다.

### 1. CARLA 학습 데이터 수집

```bash
python -m src.data.step09_dl_data_collector
python -m src.data.step09_advanced_collector
```

### 2. 데이터 준비 및 증강

```bash
python -m src.data.step20_data_augmentation
python -m src.data.step23_aihub_to_dataset
```

### 3. 모델 학습

```bash
python -m src.models.step12_train
python -m src.models.step12_advanced_train
```

### 4. 모델 변환 및 벤치마크

```bash
python -m src.benchmarks.step24_benchmark_models
python -m src.optimization.step26_export_onnx
python -m src.optimization.step27_quantize_onnx
python -m src.optimization.step27_build_tensorrt
python -m src.benchmarks.step31_grand_benchmark
```

### 5. 추론 영상 렌더링

```bash
python -m src.benchmarks.step32_make_inference_video
python -m src.benchmarks.step33
```

## 참고 사항

- CARLA 관련 스크립트는 CARLA 시뮬레이터가 실행 중이어야 합니다.
- TensorRT 관련 스크립트는 NVIDIA GPU, CUDA, TensorRT, PyCUDA, NVML 환경이 필요합니다.
- `src.utils.inference_combined`는 외부 UFLDv2 파일이 필요한 실험용 스크립트입니다.
- 첫 공개 버전에는 데이터셋 샘플을 포함하지 않습니다.

## 테스트

### 세그멘테이션 모델 벤치마크

데이터 증강 전/후 모델과 ResNet 인코더 적용 모델을 비교하는 테스트입니다.

| 모델 | 데이터 | 인코더 | 지표 | 결과 |
|---|---|---|---|---:|
| Vanilla U-Net | 원본 | Custom U-Net | mIoU | 1.34% |
| Vanilla U-Net | 증강 | Custom U-Net | mIoU | 13.63% |
| ResNet U-Net | 증강 | ResNet34 | mIoU | 20.88% |
| ResNet U-Net | 증강 | ResNet50 | mIoU | 25.79% |

### 양자화 포맷별 벤치마크

ResNet34 기반 모델을 ONNX Runtime 및 TensorRT 포맷별로 비교하는 테스트입니다.

| 런타임 | 포맷 | FPS | 지표 | 성능 |
|---|---|---|---|---:|
| ONNX Runtime | FP32 | 29.7 FPS | mIoU | 20.88% |
| ONNX Runtime | FP16 | 15.3 FPS | mIoU | 20.88% |
| ONNX Runtime | INT8 | 34.9 FPS | mIoU | 20.66% |
| TensorRT | FP16 | 155.2 FPS | mIoU | 20.88% |
| TensorRT | INT8 | 180.2 FPS | mIoU | 20.88% |
| Jetson Nano TensorRT | FP16 | 26.4 FPS | mIoU | 20.82% |

## 라이선스

이 프로젝트는 MIT License로 공개합니다. 자세한 내용은 [LICENSE](LICENSE)를 참고하세요.

외부 의존성과 데이터셋은 각각의 라이선스 및 이용 약관을 따릅니다.
