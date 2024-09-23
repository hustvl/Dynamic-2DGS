import torch
#from scene import Scene
import os
from tqdm import tqdm
import nvdiffrast.torch as dr
from utils.graphics_utils import getWorld2View2
#from plyfile import PlyData
import numpy as np
import torch
import plyfile
import math
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    SoftPhongShader,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    Materials,
    TexturesVertex,
    DirectionalLights,
)


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(
        attr.contiguous(), rast, attr_idx, rast_db=rast_db,
        diff_attrs=None if rast_db is None else 'all')

def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.nn.functional.pad(pos, pad=(0, 1), mode='constant', value=1.0)
    #posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=2)
    return torch.matmul(posw, t_mtx.t())[None, ...]


def render_mesh(viewpoint_camera, verts, faces, vertex_colors, whitebackground):
    """
    Render the mesh. 
    
    Background tensor (bg_color) must be on GPU!
    """
    
    ctx = dr.RasterizeCudaContext(device="cuda")
    
    mesh_v_pos_bxnx3 = verts
    mesh_t_pos_idx_fx3 = faces.to(torch.int32)
    mesh_v_feat_bxnxd = vertex_colors
 
    our_R = viewpoint_camera.R
    our_T = viewpoint_camera.T
    
    w2c = np.eye(4)
    w2c[:3, :3] = our_R.T
    w2c[:3, 3] = our_T
    r_mv = w2c
    
    proj = getProjectionMatrix(0.01,100,viewpoint_camera.FoVx,viewpoint_camera.FoVy).cpu().numpy()
    r_mvp = np.matmul(proj, r_mv).astype(np.float32)
    
    v_pos_clip  = transform_pos(r_mvp, mesh_v_pos_bxnx3)
    v_pos_clip = v_pos_clip.squeeze(0)
    
    # Render the image,
    # Here we only return the feature (3D location) at each pixel, which will be used as the input for neural render
    num_layers = 1
    with dr.DepthPeeler(ctx, v_pos_clip, mesh_t_pos_idx_fx3, resolution=[800, 800]) as peeler:
        for _ in range(num_layers):
            rast, db = peeler.rasterize_next_layer()
            cv_image2 = rast[...,-1:].squeeze().cpu().byte().numpy()

            mesh_v_feat_bxnxd = torch.flip(mesh_v_feat_bxnxd, dims=[2])
            output, _ = dr.interpolate(mesh_v_feat_bxnxd, rast, mesh_t_pos_idx_fx3)
            output = dr.antialias(output, rast, v_pos_clip, mesh_t_pos_idx_fx3)
            output = output[0]
            
            # Mask out the background
            mesh_v_feat_bxnxd = torch.ones(
                mesh_v_pos_bxnx3.shape, dtype=torch.float, device=v_pos_clip.device
            )
            color, _ = dr.interpolate(mesh_v_feat_bxnxd, rast, mesh_t_pos_idx_fx3)
            color = dr.antialias(color, rast, v_pos_clip, mesh_t_pos_idx_fx3)
            mask = color[0, :, :]
            #mask = torch.flip(mask, dims=[0])
            if not whitebackground:
                output[~mask.bool()] = 0
            else:
                output[~mask.bool()] = 1
            output = torch.clamp(output, 0.0, 1.0)
            
            output = output.squeeze(0)
            mesh_image = output
            
            
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": mesh_image
    }
  
    return rets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blender2p3d = torch.tensor([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1]]).float().to(device)


def mesh_shape_renderer(
    vertices: torch.Tensor, faces: torch.Tensor, viewpoint_cam
) -> torch.Tensor:
    """Render mesh shape using PyTorch3D

    Args:
        vertices (torch.Tensor): Mesh vertices [N, 3]
        faces (torch.Tensor): Mesh faces [M, 3]
        viewpoint_cam (Camera): Viewpoint camera

    Returns:
        torch.Tensor: Mesh shape image [H, W, 3]
    """
    vertices = vertices.squeeze(0)
    colors = torch.ones(1, vertices.shape[0], 3).to("cuda")
    textures = TexturesVertex(verts_features=colors)
    mesh = Meshes(
        verts=vertices.unsqueeze(0), faces=faces.unsqueeze(0), textures=textures
    )
    
    R = torch.from_numpy(viewpoint_cam.R).unsqueeze(0)
    T = torch.from_numpy(viewpoint_cam.T).unsqueeze(0)

    if True:
        fovx = viewpoint_cam.FoVx
        fovy = viewpoint_cam.FoVy
        fx = fov2focal(fovx, viewpoint_cam.image_width)
        fy = fov2focal(fovy, viewpoint_cam.image_height)
        cx = viewpoint_cam.image_width / 2
        cy = viewpoint_cam.image_height / 2
    else:
        K = torch.tensor(viewpoint_cam.K).float().to("cuda")
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

    cameras = PerspectiveCameras(
        R=R,
        T=T,
        focal_length=((fx, fy),),
        principal_point=((viewpoint_cam.image_height - cx, cy),),
        image_size=((viewpoint_cam.image_height, viewpoint_cam.image_width),),
        device=torch.device("cuda"),
        in_ndc=False,
    )
    
    W2C = getWorld2View2(viewpoint_cam.R, viewpoint_cam.T)
    C2W = np.linalg.inv(W2C)

    light_pos = (
        torch.tensor(C2W[:3, 3])
        .float()
        .to("cuda")
        .unsqueeze(0)
    )
    # Get mesh vertices rought center
    mesh_center = vertices.mean(0)
    light_dir = mesh_center - light_pos
    #light_dir = -light_dir
    lights = DirectionalLights(
        device=torch.device("cuda"),
        direction=-light_dir,
    )
    materials = Materials(
        device="cuda",
        specular_color=[[0.2, 0.2, 0.2]],
        shininess=10,  # Control the shininess for the specular component
    )
    raster_settings = RasterizationSettings(
        image_size=(viewpoint_cam.image_height, viewpoint_cam.image_width),
        blur_radius=0.0,
        faces_per_pixel=10,
        #bin_size =0,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=torch.device("cuda"),
            cameras=cameras,
            lights=lights,
            materials=materials,
        ),
    )
    mesh_img = renderer(mesh)[0][..., :3].flip(0).flip(1)
    return mesh_img