#CUDA_VISIBLE_DEVICES=3 python train_gui.py --source_path /data3/zhangshuai/SC-GS/data/jumpingjacks --model_path outputs/jumpingjacks --deform_type node --node_num 512 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800

#CUDA_VISIBLE_DEVICES=2 python train_gui.py --source_path /data3/zhangshuai/SC-GS/data/bouncingballs --model_path outputs/bouncingballs --deform_type node --node_num 512 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800


CUDA_VISIBLE_DEVICES=0 python train_gui.py --source_path /data3/zhangshuai/SC-2DGSv2/data/bouncingballs --model_path outputs/bouncingballs_0725 --deform_type mlp --node_num 512 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 1 --W 800 --H 800
CUDA_VISIBLE_DEVICES=0 python render_mesh.py --source_path /data3/zhangshuai/SC-2DGSv2/data/bouncingballs --model_path outputs/bouncingballs_0725 --deform_type mlp --node_num 512 --hyper_dim 8 --is_blender --eval --local_frame --resolution 1
CUDA_VISIBLE_DEVICES=0 python /data3/zhangshuai/SC-2DGSv2/metrics.py -m /data3/zhangshuai/SC-2DGSv2/outputs/bouncingballs_0725_mlp