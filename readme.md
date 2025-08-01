<div align="center">
<h1 align="center">
  Dynamic 2D Gaussians: Geometrically Accurate Radiance Fields for Dynamic Objects
</h1>
  
### ACMMM 2025
  
### [arXiv Paper](https://arxiv.org/abs/2409.14072)

[Shuai Zhang](https://github.com/Shuaizhang7) <sup>1\*</sup>, [Guanjun Wu](https://guanjunwu.github.io/) <sup>2\*</sup>,[Xinggang Wang](https://xwcv.github.io/) <sup>1</sup>,[Bin Feng](https://scholar.google.com/citations?user=nRc8u6gAAAAJ&hl=en&oi=ao) <sup>1</sup>,
[Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu) <sup>1,📧</sup>

<sup>1</sup> School of Electronic Information and Communications, Huazhong University of Science and Technology \
<sup>2</sup>  School of Computer Science & Technology, Huazhong University of Science and Technology 

(\* Equal contributions.📧 Corresponding author) 

</div>

---

## Abstract

Reconstructing objects and extracting high-quality surfaces play a vital role in the real world. Current 4D representations show the ability to render high-quality novel views for dynamic objects but cannot reconstruct high-quality meshes due to their implicit or geometrically inaccurate representations. In this paper, we propose a novel representation that can reconstruct accurate meshes from sparse image input, named Dynamic 2D Gaussians (D-2DGS). We adopt 2D Gaussians for basic geometry representation and use sparse-controlled points to capture 2D Gaussian's deformation. By extracting the object mask from the rendered high-quality image and masking the rendered depth map, a high-quality dynamic mesh sequence of the object can be extracted. Experiments demonstrate that our D-2DGS is outstanding in reconstructing high-quality meshes from sparse input.

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

- 2025-07-05: Accepted by ACMMM 2025

- 2024-09-24: Release code

## Installation

```bash
git clone --recursive git@github.com:hustvl/Dynamic-2DGS.git
cd Dynamic-2DGS
conda create --name dynamic-2dgs python=3.8.0
conda activate dynamic-2dgs

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

pip install ./submodules/diff-surfel-rasterization
pip install ./submodules/simple-knn
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

pip install -r requirements.txt
```

## Data
We use the following dataset:
* [D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html): dynamic scenes of synthetic objects ([download](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?e=1&dl=0))
* [DG-Mesh](https://github.com/Isabella98Liu/DG-Mesh?tab=readme-ov-file): dynamic scenes of synthetic objects ([download](https://drive.google.com/file/d/1yBga2DsIKG6zQK9V2WextewvhaV8Xho3/view))

## Run

### Training

```bash
CUDA_VISIBLE_DEVICES=0 python train_gui.py --source_path YOUR/PATH/TO/DATASET/jumpingjacks --model_path outputs/jumpingjacks --deform_type node  --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 1 --W 800 --H 800
```

### Rendering
Rendering test views:
```bash
CUDA_VISIBLE_DEVICES=0 python render_mesh.py --source_path YOUR/PATH/TO/DATASET/jumpingjacks --model_path outputs/jumpingjacks --deform_type node --hyper_dim 8 --is_blender --eval --local_frame --resolution 1
```

Render a view track:
```bash
CUDA_VISIBLE_DEVICES=0 python render_mesh_trajectory.py --source_path YOUR/PATH/TO/DATASET/jumpingjacks --model_path outputs/jumpingjacks --deform_type node --hyper_dim 8 --is_blender --eval --local_frame --resolution 1
```

### Evaluation
Evaluating the rendered image:
```bash
CUDA_VISIBLE_DEVICES=0 python metrics.py -m outputs/jumpingjacks
```

To evaluate the quality of the mesh, you can refer to the [DG-Mesh](https://github.com/Isabella98Liu/DG-Mesh) repository.

### 3D Printing
Reconstruct the mesh through our model and 3D print it:
<div align="center">
<img src="./assets/print.jpg" width="45%">
<img src="./assets/trex.gif" width="35%">
</div>

## Citation
If you find our work useful, please consider citing:
```BibTeX
@misc{zhang2024dynamic2dgaussiansgeometrically,
      title={Dynamic 2D Gaussians: Geometrically accurate radiance fields for dynamic objects}, 
      author={Shuai Zhang and Guanjun Wu and Xinggang Wang and Bin Feng and Wenyu Liu},
      year={2024},
      eprint={2409.14072},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.14072}, 
}
```

## Acknowledgement

* Our work is developed on the basis of [DG-Mesh](https://github.com/Isabella98Liu/DG-Mesh), [SCGS](https://github.com/yihua7/SC-GS) and [2DGS](https://github.com/hbb1/2d-gaussian-splatting/tree/main), thanks to these great works.

```
@misc{liu2024dynamic,
 title={Dynamic Gaussians Mesh: Consistent Mesh Reconstruction from Monocular Videos}, 
 author={Isabella Liu and Hao Su and Xiaolong Wang},
 year={2024},
 eprint={2404.12379},
 archivePrefix={arXiv},
 primaryClass={cs.CV}
}

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



