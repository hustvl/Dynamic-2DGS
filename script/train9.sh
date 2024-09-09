#CUDA_VISIBLE_DEVICES=3 python train_gui.py --source_path /data3/zhangshuai/SC-GS/data/jumpingjacks --model_path outputs/jumpingjacks --deform_type node --node_num 512 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800

#CUDA_VISIBLE_DEVICES=2 python train_gui.py --source_path /data3/zhangshuai/SC-GS/data/bouncingballs --model_path outputs/bouncingballs --deform_type node --node_num 512 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800
dataname='hellwarrior'
datetime='0811'
d_type=node



#CUDA_VISIBLE_DEVICES=2 python train_gui.py --source_path /data3/zhangshuai/SC-2DGSv2/data/$dataname --model_path outputs/${dataname}_${datetime} --deform_type $d_type  --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 1 --W 800 --H 800
CUDA_VISIBLE_DEVICES=2 python render_mesh2.py --source_path /data3/zhangshuai/SC-2DGSv2/data/$dataname --model_path outputs/${dataname}_${datetime} --deform_type $d_type --hyper_dim 8 --is_blender --eval --local_frame --resolution 1
#CUDA_VISIBLE_DEVICES=2 python /data3/zhangshuai/SC-2DGSv2/metrics.py -m /data3/zhangshuai/SC-2DGSv2/outputs/${dataname}_${datetime}_$d_type

#CUDA_VISIBLE_DEVICES=1 python render_mesh.py --config /data3/zhangshuai/SC-2DGSv2/configs/d-nerf/bouncingballs_mesh.yaml