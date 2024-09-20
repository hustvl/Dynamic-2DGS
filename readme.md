
<div align="center">
  <img src="./assets/bouncingballs.gif" width="24.5%">
  <img src="./assets/horse.gif" width="24.5%">
  <img src="./assets/jumpingjacks.gif" width="24.5%">
  <img src="./assets/standup.gif" width="24.5%">

</div>
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


## Updates

### 2024-03-17:


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

* This framework has been adapted from the notable [Deformable 3D Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians), an excellent and pioneering work by [Ziyi Yang](https://github.com/ingra14m).
```
@article{yang2023deformable3dgs,
    title={Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction},
    author={Yang, Ziyi and Gao, Xinyu and Zhou, Wen and Jiao, Shaohui and Zhang, Yuqing and Jin, Xiaogang},
    journal={arXiv preprint arXiv:2309.13101},
    year={2023}
}
```

* Credits to authors of [3D Gaussians](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) for their excellent code.
```
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

## Citing
If you find our work useful, please consider citing:
```BibTeX
@article{huang2023sc,
  title={SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes},
  author={Huang, Yi-Hua and Sun, Yang-Tian and Yang, Ziyi and Lyu, Xiaoyang and Cao, Yan-Pei and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2312.14937},
  year={2023}
}
```
