import cv2

import numpy as np



output_path = '/data3/zhangshuai/SC-2DGSv2/outputs/lego_result/'
d3dgs_i_path = output_path+'image/d3dgs/v00018_zoomed.png'
dgmesh_i_path = output_path+'image/dgmesh/v00018_zoomed.png'
ours_i_path = output_path+'image/ours/v00018_zoomed.png'
scgs_i_path = output_path+'image/scgs/v00018_zoomed.png'

d3dgs_m_path = output_path+'mesh/d3dgs/v00018_zoomed.png'
dgmesh_m_path = output_path+'mesh/dgmesh/v00018_zoomed.png'
ours_m_path = output_path+'mesh/ours/v00018_zoomed.png'
scgs_m_path = output_path+'mesh/scgs/v00018_zoomed.png'

gt_path = output_path+'gt/v00018_zoomed.png'
gt_mesh_path =None
#gt_mesh_path = output_path+'gt_mesh/v00002_zoomed.png'



def margeimage(dgmesh_i,d3dgs_i,scgs_i,ours_i,dgmesh_m,d3dgs_m,scgs_m,ours_m,gt,gt_m,output_path):
    width1 = gt.shape[1]
    heigth1 = gt.shape[0]
    all_width = 5*width1
    all_heigth = 2*heigth1
    new_image = np.zeros((all_heigth, all_width, 3), dtype=np.uint8)
    new_image.fill(255)
    new_image[:heigth1,:width1] = dgmesh_i
    new_image[:heigth1,width1:2*width1] = d3dgs_i
    new_image[:heigth1,2*width1:3*width1] = scgs_i
    new_image[:heigth1,3*width1:4*width1] = ours_i
    new_image[:heigth1,4*width1:5*width1] = gt
    new_image[heigth1:,:width1] = dgmesh_m
    new_image[heigth1:,width1:2*width1] = d3dgs_m
    new_image[heigth1:,2*width1:3*width1] = scgs_m
    new_image[heigth1:,3*width1:4*width1] = ours_m
    if gt_mesh_path != None:
        new_image[heigth1:,4*width1:5*width1] = gt_m

    cv2.imwrite(output_path+"result.png", new_image)
    

#for i in ['d3dgs_i','dgmesh_i','ours_i','scgs_i','d3dgs_m','dgmesh_m','ours_m','scgs_m','gt']:

d3dgs_i = cv2.imread(d3dgs_i_path)
dgmesh_i = cv2.imread(dgmesh_i_path)
ours_i = cv2.imread(ours_i_path)
scgs_i = cv2.imread(scgs_i_path)

d3dgs_m = cv2.imread(d3dgs_m_path)
dgmesh_m = cv2.imread(dgmesh_m_path)
ours_m = cv2.imread(ours_m_path)
scgs_m = cv2.imread(scgs_m_path)

gt = cv2.imread(gt_path)
gt_m =None
if gt_mesh_path != None:
    gt_m = cv2.imread(gt_mesh_path)


margeimage(dgmesh_i,d3dgs_i,scgs_i,ours_i,dgmesh_m,d3dgs_m,scgs_m,ours_m,gt,gt_m,output_path)

