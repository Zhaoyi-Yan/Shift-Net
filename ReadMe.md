# Shift-Net
**This repository is deprecated** ! Please refer to our pytorch version [Shift-Net_pytorch](https://github.com/Zhaoyi-Yan/Shift-Net_pytorch). That is much faster than this repository. As some code in this repository is implemented using `for-loop`, while the code of pytorch version [Shift-Net_pytorch](https://github.com/Zhaoyi-Yan/Shift-Net_pytorch) is fully-implemented parallelly.



If you find this paper useful, please cite:
```
@InProceedings{Yan_2018_Shift,
author = {Yan, Zhaoyi and Li, Xiaoming and Li, Mu and Zuo, Wangmeng and Shan, Shiguang},
title = {Shift-Net: Image Inpainting via Deep Feature Rearrangement},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
}
```

## Acknowledgments
We benefit a lot from [pix2pix](https://github.com/phillipi/pix2pix) and [DCGAN](https://github.com/soumith/dcgan.torch). The data loader is modified from [pix2pix](https://github.com/phillipi/pix2pix) and the implemetation of Instance Normalization borrows from [Instance Normalization](https://github.com/DmitryUlyanov/texture_nets/blob/master/InstanceNormalization.lua). The shift operation is inspired by [style-swap](https://github.com/rtqichen/style-swap).
