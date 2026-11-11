# RedReg

## Redundancy-Aware Regulation for Balanced Multimodal Information Acquisition

RedReg is a PyTorch implementation of redundancy-aware regularization for audio-visual emotion recognition on CREMA-D. This repository contains the RedReg implementation, including the Joint configuration and the Star configuration corresponding to RedReg*. The implementation is based on the public [InfoReg](https://github.com/GeWu-Lab/InfoReg_CVPR2025) audio-visual training code and was tested with two NVIDIA RTX 3090 GPUs.

### Installing

The RedReg environment uses Python 3.8, PyTorch 1.8.0, and torchvision 0.9.0.

```bash
conda create -n redreg python=3.8 -y
conda activate redreg
python -m pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -r requirements.txt
```

### Downloading CREMA-D

Download the original CREMA-D audio and video files from the [CREMA-D repository](https://github.com/CheyneyComputerScience/CREMA-D). The media are not included in this repository.

Place the files in the following structure:

```text
datasets/CREMA-D/
├── AudioWAV/
└── VideoFlash/
```
### Preparing features

```bash
python scripts/prepare_cremad.py \
  --dataset-root datasets/CREMA-D --workers 8

python scripts/build_audio_cache.py \
  --audio-path datasets/CREMA-D/AudioWAV \
  --folder datasets/CREMA-D/RedRegAudioCache --workers 8
```

### Training

```bash
REDREG_GPUS=0,1 bash scripts/run.sh \
  --config configs/cremad_star.json
```

### Evaluation

Evaluate a fixed-epoch checkpoint with the preserved RedReg test-frame manifest:

```bash
REDREG_GPUS=0,1 bash scripts/run.sh \
  --config configs/cremad_star.json \
  --run-name RedReg_star_evaluation \
  --evaluate runs/RedReg_star_seed0/checkpoints/epoch_050.pt
```

### Acknowledgement

We thank the authors of [InfoReg](https://github.com/GeWu-Lab/InfoReg_CVPR2025) and [OGM-GE](https://github.com/GeWu-Lab/OGM-GE_CVPR2022) for their public audio-visual code and data-processing components. The inherited MIT notice is preserved in [LICENSE](LICENSE). Please follow the original CREMA-D dataset license when obtaining and redistributing the media.
