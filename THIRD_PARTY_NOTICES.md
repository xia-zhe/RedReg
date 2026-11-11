# Third-party notices

RedReg uses the audio/visual ResNet, CREMA-D dataset loader, initialization
utilities and baseline initialization structure from
[InfoReg](https://github.com/GeWu-Lab/InfoReg_CVPR2025), commit
`2e61ffb9901bcd849d62b60bb86f2374789cedaf`.
The inherited code includes the MIT copyright notice for GeWu-Lab (2022),
preserved verbatim in `LICENSE`, and derives from
[OGM-GE](https://github.com/GeWu-Lab/OGM-GE_CVPR2022).

The Gaussian bottleneck, redundancy monitor, prediction consistency gate,
training integration and packaging are an independent implementation of RedReg.
This repository is not the paper authors' official implementation.

CREMA-D media are not distributed here. Obtain them from the
[CREMA-D dataset repository](https://github.com/CheyneyComputerScience/CREMA-D)
and retain the dataset's own license. The files under `data/CREMAD` contain
split labels and frame filenames only; they contain no audio or images.
