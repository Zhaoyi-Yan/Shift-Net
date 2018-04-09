
# Shift-Net

## Getting Started
We expect you have an nvidia GPU and have install CUDA.
The code does not support running on CPU for now.
### Installation
- Install torch and dependencies from https://github.com/torch/distro
- Install torch packages `nngraph`, `cudnn`, `display`
```bash
luarocks install nngraph
luarocks install cudnn
luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec
```
- Clone this repo:
```bash
git clone https://github.com/Zhaoyi-Yan/Shift-Net
cd Shift-Net
```

### Download pre-trained model
```bash
bash download_model.sh
```
The model will be download and unzipped.

### Train
- Download your own dataset.

- Change the options in `train.lua` according to your path of dataset.
Normally, you should at least specify three options.
They are `DATA_ROOT`, `phase` and `name`.

For example:

`DATA_ROOT`: `./datasets/Paris_StreetView_Dataset/`

`phase`:     `paris_train`

`name`:      `paris_train_shiftNet`

This means that the training images are under the folder of `./datasets/Paris_StreetView_Dataset/paris_train/`.
As for `name`, it gives your experiment a name, e.g., `paris_train_shiftNet`. When training, the checkpoints are stored under the folder
`./checkpoints/paris_train_shiftNet/`.



- Train a model:
```bash
th train.lua
```

- Display the temporary results on the browser.
Set `display = 1`, and then open another console, 
```bash
th -ldisplay.start
```
- Open this URL in your browser: [http://localhost:8000](http://localhost:8000)

### Test
Before test, you should change `DATA_ROOT`, `phase`, `name`, `checkpoint_dir` and `which_epoch`.
For example, if you want to test the 30-th epoch of your trained model, then
`DATA_ROOT`:    `./datasets/Paris_StreetView_Dataset/`

`phase`:        `paris_train`

`name`:         `paris_train_shiftNet`

`checkpoint_dir`:`./checkpoints/`

`which_epoch`: `'30'`

The first two options determine where the dataset is, and the rest define the folder where the model is stored.
- Finally, test the model:
```bash
th test.lua
```


## Acknowledgments
We benefit a lot from [pix2pix](https://github.com/phillipi/pix2pix) and [DCGAN](https://github.com/soumith/dcgan.torch). The data loader is modified from [pix2pix](https://github.com/phillipi/pix2pix) and the implemetation of Instance Normalization borrows form [Instance Normalization](https://github.com/DmitryUlyanov/texture_nets/blob/master/InstanceNormalization.lua). The shift operation is inspired by [style-swap](https://github.com/rtqichen/style-swap).