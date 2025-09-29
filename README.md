# RedReg
## Paper title:
Adaptive Redundancy Regulation for Balanced Multimodal Information Refinement

## Authors:
Zhe Yang, Wenrui Li, Hongtao Chen, Penghong Wang, Ruiqin Xiong, Xiaopeng Fan


## Abstract
Multimodal learning aims to improve performance by leveraging data from multiple sources. During joint multimodal training, due to modality bias, the advantaged modality often dominates backpropagation, leading to imbalanced optimization. Existing methods still face two problems: First, the long-term dominance of the dominant modality weakens representation-output coupling in the late stages of training, resulting in the accumulation of redundant information. Second, previous methods often directly and uniformly adjust the gradients of the advantaged modality, ignoring the semantics and directionality between modalities. To address these limitations, we propose Adaptive Redundancy Regulation for Balanced Multimodal Information Refinement (RedReg), which is inspired by information bottleneck principle. Specifically, we construct a redundancy phase monitor that uses a joint criterion of effective gain growth rate and redundancy to trigger intervention only when redundancy is high. Furthermore, we design a co-information gating mechanism to estimate the contribution of the current dominant modality based on cross-modal semantics. When the task primarily relies on a single modality, the suppression term is automatically disabled to preserve modality-specific information. Finally, we project the gradient of the dominant modality onto the orthogonal complement of the joint multimodal gradient subspace and suppress the gradient according to redundancy. Experiments show that our method demonstrates superiority
among current major methods in most scenarios. Ablation experiments verify the effectiveness of our method.
For more details of our paper, please refer to our paper.


## Overview of RedReg

<div align="center">    
    <img src="pictures/fig2.pdf" style="width: 90%" />
</div>

## Dataset 

The original datasets can be found:
[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D),
[Kinetics-Sounds](https://github.com/cvdfoundation/kinetics-dataset),
[CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)

Data processing follows [OGM](https://github.com/GeWu-Lab/OGM-GE_CVPR2022).


## Run the code
After data processing, you can simply run the code by:
``` 
python main_RedReg.py 
```
### Dowloading pre-trained models
The pre-trained weights will be released soon.

## Acknowledgement
We appreciate the code provided by [InfoReg](https://github.com/GeWu-Lab/InfoReg_CVPR2025), which is very helpful to our research.
