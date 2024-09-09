from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr,get_psnr
from argparse import ArgumentParser
import cv2

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        if len(fname.split('.')[0]) <= 5:
            render = Image.open(renders_dir +'/'+ fname)
            if len(fname.split('.')[0])<5:
                fname = fname.split('.')[0].zfill(5)+'.png'
            gt = Image.open(gt_dir +'/'+ fname)
            renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
            gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
            image_names.append(fname)
    print(image_names)
    return renders, gts, image_names




def metrics(render_path,gt_path,savepath,name):
    imagelist = os.listdir(gt_path)
    result = {}
    #renders, gts, image_names = readImages(render_path, gt_path)

    renders, gts, image_names = readImages(render_path, gt_path)

    ssims = []
    psnrs = []
    lpipss = []

    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        ssims.append(ssim(renders[idx], gts[idx]))
        psnrs.append(get_psnr(renders[idx], gts[idx]))
        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
    print("")

    result.update({"SSIM": torch.tensor(ssims).mean().item(),
                                            "PSNR": torch.tensor(psnrs).mean().item(),
                                            "LPIPS": torch.tensor(lpipss).mean().item()})
        
    with open(savepath + f"/{name}_results.json", 'w') as fp:
        json.dump(result, fp, indent=True)
        


'''     
for name in ['d3dgs','dgmesh','ours','scgs']:      
    gt_path = '/data3/zhangshuai/SC-2DGSv2/outputs/hellwarrior_result/gt'
    render_path = f'/data3/zhangshuai/SC-2DGSv2/outputs/hellwarrior_result/image/{name}'
    savepath =  '/data3/zhangshuai/SC-2DGSv2/outputs/hellwarrior_result/image'
    metrics(render_path,gt_path,savepath,name)
'''

gt_path = '/data3/zhangshuai/SC-2DGSv2/outputs/torus2sphere_0801_node/test/ours_70000/gt_w'
render_path = '/data3/zhangshuai/SC-2DGSv2/outputs/torus2sphere_0801_node/mesh_image'
savepath =  '/data3/zhangshuai/SC-2DGSv2/outputs/torus2sphere_0801_node'
name= 'mesh_render'

metrics(render_path,gt_path,savepath,name)