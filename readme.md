<div align="center">
<h1 align="center">
  Dynamic 2D Gaussians: Geometrically accurate radiance fields for dynamic objects
</h1>

### [arXiv Paper](https://arxiv.org/abs/2403.19586)

[Shuai Zhang](https://github.com/Shuaizhang7) <sup>1\*</sup>, [Guanjun Wu](https://guanjunwu.github.io/) <sup>2\*</sup>,[Xinggang Wang](https://xwcv.github.io/) <sup>1</sup>,[Bin Feng]() <sup>1</sup>,
[Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu) <sup>1,📧</sup>

<sup>1</sup> School of Electronic Information and Communications, Huazhong University of Science and Technology \
<sup>2</sup>  School of Computer Science &Technology, Huazhong University of Science and Technology 

(\* equal contribution,📧 corresponding author) 

</div>

---

## Abstract
<div align="center">
  <img src="./assets/teaser.png" width="100%" height="100%">
</div>

*Framework of our D-2DGS. Sparse points are bonded with canonical 2D Gaussians. Deformation networks are used to predict each sparse control point's control signals given any timestamp. The image and depth are rendered by deformed 2D Gaussians with alpha blending. To get high-quality meshes, depth images are filtered by rendered images with RGB mask, and then TSDF is applied on multiview depth images and RGB images.*

## Demo

<div align="center">
  <img src="./assets/bouncingballs.gif" width="24.5%">
  <img src="./assets/horse.gif" width="24.5%">
  <img src="./assets/jumpingjacks.gif" width="24.5%">
  <img src="./assets/standup.gif" width="24.5%">

</div>

## Updates

### 2024-09-20: Upload code


## Install

```bash
git clone https://github.com/yihua7/SC-GS --recursive
cd SC-GS

pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
pip install ./submodules/diff-gaussian-rasterization

# simple-knn
pip install ./submodules/simple-knn
```

## Run

### Train

```bash
CUDA_VISIBLE_DEVICES=0 python train_gui.py --source_path YOUR/PATH/TO/DATASET/jumpingjacks --model_path outputs/jumpingjacks --deform_type node  --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 1 --W 800 --H 800
```

### Evalualuate

```bash
CUDA_VISIBLE_DEVICES=0 python render_mesh.py --source_path YOUR/PATH/TO/DATASET/jumpingjacks --model_path outputs/jumpingjacks --deform_type node --hyper_dim 8 --is_blender --eval --local_frame --resolution 1
```


## Acknowledgement

* Our work is developed on the basis of [SCGS](https://github.com/yihua7/SC-GS) and [2DGS](https://github.com/hbb1/2d-gaussian-splatting/tree/main), thanks to these two great works.

```
@article{huang2023sc,
  title={SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes},
  author={Huang, Yi-Hua and Sun, Yang-Tian and Yang, Ziyi and Lyu, Xiaoyang and Cao, Yan-Pei and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2312.14937},
  year={2023}
}

@inproceedings{Huang2DGS2024,
    title={2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
    author={Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
    publisher = {Association for Computing Machinery},
    booktitle = {SIGGRAPH 2024 Conference Papers},
    year      = {2024},
    doi       = {10.1145/3641519.3657428}
}
```


## Citing
If you find our work useful, please consider citing:
```BibTeX

```
