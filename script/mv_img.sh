name=hellwarrior_result
our_name=hellwarrior_0811_node
scgs_name=hellwarrior_0811_node
deform_name=hellwarrior_w
dgm_path=/data3/zhangshuai/DG-Mesh/outputs/d-nerf/hellwarrior/rendering-traj-hellwarrior-2024-08-12_20-57-39


cp -r $dgm_path/images/image/* /data3/zhangshuai/SC-2DGSv2/outputs/$name/image/dgmesh
cp -r $dgm_path/images/mesh/* /data3/zhangshuai/SC-2DGSv2/outputs/$name/mesh/dgmesh

cp -r /data3/zhangshuai/SC-2DGSv2/outputs/hellwarrior_0811_node/test/ours_40000/gt/* /data3/zhangshuai/SC-2DGSv2/outputs/$name/gt
cp -r /data3/zhangshuai/SC-2DGSv2/outputs/$our_name/mesh_shape_gt/* /data3/zhangshuai/SC-2DGSv2/outputs/$name/gt_mesh
cp -r /data3/zhangshuai/SC-2DGSv2/outputs/$our_name/mesh_shape/* /data3/zhangshuai/SC-2DGSv2/outputs/$name/mesh/ours
cp -r /data3/zhangshuai/SC-2DGSv2/outputs/$our_name/mesh_image/* /data3/zhangshuai/SC-2DGSv2/outputs/$name/image/ours

cp -r /data3/zhangshuai/SC-2DGS/outputs/$scgs_name/mesh_shape/* /data3/zhangshuai/SC-2DGSv2/outputs/$name/mesh/scgs
cp -r /data3/zhangshuai/SC-2DGS/outputs/$scgs_name/mesh_image/* /data3/zhangshuai/SC-2DGSv2/outputs/$name/image/scgs

scp -r -P 2025 zhangshuai@115.156.156.112:/data5/zhangshuai/Deformable-3D-Gaussians/output/$deform_name/mesh_shape/* /data3/zhangshuai/SC-2DGSv2/outputs/$name/mesh/d3dgs
scp -r -P 2025 zhangshuai@115.156.156.112:/data5/zhangshuai/Deformable-3D-Gaussians/output/$deform_name/mesh_image/* /data3/zhangshuai/SC-2DGSv2/outputs/$name/image/d3dgs

