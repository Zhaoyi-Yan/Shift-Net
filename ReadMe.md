# Architecutre
<img src="architecture.png" width="1000"/> 

# Shift layer
<img src="shift_layer.png" width="800"/> 

# Shift-Net
Please refer to our code [Shift-Net_pytorch](https://github.com/Zhaoyi-Yan/Shift-Net_pytorch) based on Pytorch. It is much faster than this repository. As some code in this repository is implemented using `for-loop`, while the code of pytorch version [Shift-Net_pytorch](https://github.com/Zhaoyi-Yan/Shift-Net_pytorch) is fully-implemented parallelly.

This repository is desperated! Please refer to our pytorch version [Shift-Net_pytorch](https://github.com/Zhaoyi-Yan/Shift-Net_pytorch).

<img src="./imgs/01_in.png" width="210"/> <img src="./imgs/01_out.png" width="210"/>
<img src="./imgs/02_in.png" width="210"/> <img src="./imgs/02_out.png" width="210"/>
<img src="./imgs/03_in.png" width="210"/> <img src="./imgs/03_out.png" width="210"/>
<img src="./imgs/04_in.png" width="210"/> <img src="./imgs/04_out.png" width="210"/>
<img src="./imgs/06_in.png" width="210"/> <img src="./imgs/06_out.png" width="210"/>
<img src="./imgs/10_in.png" width="210"/> <img src="./imgs/10_out.png" width="210"/>
<img src="./imgs/r_01_in.png" width="210"/> <img src="./imgs/r_01_out.png" width="210"/>
<img src="./imgs/r_02_in.png" width="210"/> <img src="./imgs/r_02_out.png" width="210"/>
<img src="./imgs/r_03_in.png" width="210"/> <img src="./imgs/r_03_out.png" width="210"/>
<img src="./imgs/r_04_in.png" width="210"/> <img src="./imgs/r_04_out.png" width="210"/>


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
