dataname='torus2sphere'
datetime='0801'
d_type=node

#CUDA_VISIBLE_DEVICES=1 python train_gui.py --source_path /data3/zhangshuai/SC-2DGSv2/data/dgmesh/$dataname --model_path outputs/${dataname}_${datetime} --deform_type $d_type  --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 1 --W 800 --H 800
CUDA_VISIBLE_DEVICES=1 python render_mesh.py --source_path /data3/zhangshuai/SC-2DGSv2/data/dgmesh/$dataname --model_path outputs/${dataname}_${datetime} --deform_type $d_type --hyper_dim 8 --is_blender --eval --local_frame --resolution 1
