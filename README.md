# Edge Autonomous Driving with CARLA

[한국어 README](README.ko.md)

Sim2Real autonomous-driving perception project built around CARLA simulation,
U-Net-based lane segmentation, YOLO object detection, and ONNX/TensorRT edge
inference experiments.

The project explores whether perception models trained on simulated driving
data can be converted, optimized, and evaluated for lower-cost edge deployment.

## Features

- CARLA RGB and semantic camera data collection
- Classical lane perception with IPM, masks, and polynomial fitting
- Binary and multi-class U-Net training pipelines
- ResNet34/ResNet50 segmentation models with `segmentation_models_pytorch`
- YOLO-based vehicle, pedestrian, and traffic-light detection experiments
- Sensor-fusion driving demos with ACC and manual override
- Real-road and AI-Hub dataset conversion scripts
- PyTorch to ONNX export
- ONNX FP16/INT8 and TensorRT FP16/INT8 benchmark scripts
- Inference video rendering utilities

## Project Structure

```text
.
├── src/
│   ├── config.py          # Shared paths, CARLA connection, model paths
│   ├── data/              # Data collection, conversion, augmentation
│   ├── models/            # Dataset classes, model definitions, training
│   ├── driving/           # CARLA driving, lane tracking, sensor fusion
│   ├── optimization/      # ONNX export, ONNX quantization, TensorRT build
│   ├── benchmarks/        # Model benchmarks and video rendering
│   └── utils/             # Environment checks and debugging helpers
├── docs/
│   └── project_summary.md
├── carla_yolo.yaml
├── carla_aug_yolo.yaml
├── requirements.txt
└── README.md
```

## Environment

The original experiments used Python 3.10 with CARLA, PyTorch, OpenCV,
Ultralytics YOLO, ONNX Runtime, TensorRT, PyCUDA, and NVIDIA GPU tooling.

```bash
conda activate carla
python --version
```

When PowerShell does not activate the environment correctly, use:

```bash
conda run -n carla python --version
```

## Configuration

Shared project settings are defined in `src/config.py`.

For machine-specific overrides, create `src/local_config.py` and redefine only
the values that need to change. This local override file is ignored by Git.

## Example Workflow

Run commands from the project root.

### 1. Collect CARLA Training Data

```bash
python -m src.data.step09_dl_data_collector
python -m src.data.step09_advanced_collector
```

### 2. Prepare and Augment Datasets

```bash
python -m src.data.step20_data_augmentation
python -m src.data.step23_aihub_to_dataset
```

### 3. Train Models

```bash
python -m src.models.step12_train
python -m src.models.step12_advanced_train
```

### 4. Export and Benchmark

```bash
python -m src.benchmarks.step24_benchmark_models
python -m src.optimization.step26_export_onnx
python -m src.optimization.step27_quantize_onnx
python -m src.optimization.step27_build_tensorrt
python -m src.benchmarks.step31_grand_benchmark
```

### 5. Render Inference Videos

```bash
python -m src.benchmarks.step32_make_inference_video
python -m src.benchmarks.step33
```

## Notes

- CARLA scripts require a running CARLA simulator.
- TensorRT scripts require a compatible NVIDIA GPU, CUDA, TensorRT, PyCUDA, and NVML setup.
- `src.utils.inference_combined` is an experimental UFLDv2 integration script and requires external UFLDv2 files that are not included here.
- Dataset samples are not included in the first public release.

## Reported Results

Values below come from the final presentation materials.

| Item | Value |
|---|---:|
| ResNet34 Aug mIoU | 20.88% |
| ResNet50 Aug mIoU | 25.79% |
| ONNX FP32 | 29.7 FPS |
| ONNX FP16 | 15.3 FPS |
| ONNX INT8 | 34.9 FPS |
| TensorRT FP16 | 155.2 FPS |
| TensorRT INT8 | 180.2 FPS |
| Jetson Nano TensorRT FP16 | 26.4 FPS |

## License

No project license has been selected yet. Choose a license before presenting the
repository as fully open source.
