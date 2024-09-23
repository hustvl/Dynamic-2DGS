dataname='torus2sphere'
datetime='0801'
d_type=node

CUDA_VISIBLE_DEVICES=1 python render_mesh.py --source_path ./data/dgmesh/$dataname --model_path outputs/${dataname}_${datetime} --deform_type $d_type --hyper_dim 8 --is_blender --eval --local_frame --resolution 1
