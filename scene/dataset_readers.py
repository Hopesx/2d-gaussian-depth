#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from random import sample

from PIL import Image
import cv2
from typing import NamedTuple
from scipy.ndimage import zoom

from numpy.core.numeric import newaxis

from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import matplotlib.pyplot as plt
from SuperGlue.get_kpts_matches import get_kpts_matches

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str



def get_ray(data_folder, sample_rate=1, kpts=None):
    poses_arr = np.load(os.path.join(data_folder, 'poses_bounds.npy'))
    # poses = poses_arr[:, :-2].reshape([-1, 3, 5])
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    # print("poses", poses[0])
    origins = []
    directions = []
    final_origins = np.zeros((0, 3))
    final_directions = np.zeros((0, 3))
    center_directions = []
    for i in range(poses.shape[0]):
    # for i in range(2):
        hwf = poses[i, :3, -1]  # 取出前三行最后一列元素
        H, W, focal = hwf
        if W > 1600:
            # 计算缩放比例
            scale_factor = 1600 / W
            (W, H) = (int(W * scale_factor), int(H * scale_factor))
        origin, direction = get_rays_np(int(H), int(W), focal, poses[i, :3, :4])
        center_directions.append(direction[int(H / 2), int(W / 2)])
        sample_origin = origin[::sample_rate, ::sample_rate, :]
        sample_direction = direction[::sample_rate, ::sample_rate, :]
        if kpts is not None:
            kpts = kpts.astype(int)
            kpt_origin = origin[kpts[i, :, 1], kpts[i, :, 0], :]
            kpt_direction = direction[kpts[i, :, 1], kpts[i, :, 0], :]
            sample_origin = np.vstack((sample_origin.reshape(-1, 3), kpt_origin))
            sample_direction = np.vstack((sample_direction.reshape(-1, 3), kpt_direction))
        final_origins = np.vstack((final_origins, sample_origin.reshape(-1, 3)))
        final_directions = np.vstack((final_directions, sample_direction.reshape(-1, 3)))
        origins.append(origin)
        directions.append(direction)
    return final_origins, final_directions, np.array(center_directions), origins, directions


def get_rays_np(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -np.ones_like(i)], -1)
    # print("dirs", dirs)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def read_depth(depth_image_path):
    # 使用字符串格式化将整数转换为两位数的字符串，例如 1 -> '01'
    # image_id_str = f"{image_id + 1:02d}"
    # depthimg_files = os.listdir(datafolder)
    # depthimg_files.sort()
    #
    # # 构建深度图像文件路径
    # depth_image_path = os.path.join(datafolder, "depth", f"{image_id_str}.png")

    # 读取深度图像
    depth_image = cv2.imread(depth_image_path, cv2.COLOR_BGR2GRAY)

    # gray_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    if depth_image is None:
        raise FileNotFoundError(f"Depth image '{depth_image_path}' not found.")
    # print("原始：", depth_image)
    # 检查宽度是否大于 1600
    if depth_image.shape[1] > 1600:
        # 计算缩放比例
        scale_factor = 1600 / depth_image.shape[1]

        # 缩小图像
        new_shape = (int(depth_image.shape[1] * scale_factor), int(depth_image.shape[0] * scale_factor))
        resized_depth_image = cv2.resize(depth_image, new_shape, interpolation=cv2.INTER_LINEAR)
    else:
        resized_depth_image = depth_image

    # 转换为浮点数类型的数组
    depth_data = resized_depth_image.astype(np.float32) / 255.
    # print("/255.", depth_data)
    depth_data_norm = (depth_data - depth_data.min()) / (depth_data.max() - depth_data.min())
    depth_data = 1 - depth_data_norm + 0.5
    # depth_data = np.arctan(depth_data * 4)
    # depth_data = (depth_data - depth_data.min()) / (depth_data.max() - depth_data.min())
    # print("norm:", depth_data)
    # depth_data = 1 / (depth_data / depth_data.max())
    return depth_data[:, :, 0]


def read_depth_from_txt(filepath):
    # 从 TXT 文件读取矩阵数据
    depth = np.loadtxt(filepath)
    if depth.shape[1] > 1600:
        # 计算缩放比例
        scale_factor = 1600 / depth.shape[1]
        depth = zoom(depth, (scale_factor, scale_factor),order=1)
    return depth

def get_depths_rays(data_folder, sample_rate, keypoints_num=200, superglue='indoor'):
    """
    读取深度图像
    """
    # 均匀采样
    print("get_depths_rays from ", data_folder)
    depth_map = np.empty(0)
    rgb_image = np.empty(0)
    img_path = os.path.join(data_folder, "images")
    img_files = os.listdir(img_path)
    img_files.sort()
    depth_path = os.path.join(data_folder, "depth/images")
    depth_files = os.listdir(depth_path)
    depth_files.sort()
    all_depths = []
    all_rgbs = []
    for i in range(len(img_files)):
        # 读取rgb图像
        imgfile = os.path.join(img_path, img_files[i])
        # img = Image.open(imgfile)
        img = cv2.imread(imgfile)
        shape = img.shape
        if img.shape[1] > 1600:
            # 计算缩放比例
            scale_factor = 1600 / img.shape[1]
            # 缩小图像
            new_shape = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
            img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb / 255.
        all_rgbs.append(img_rgb)
        img_rgb = img_rgb[::sample_rate, ::sample_rate, :]

        # 使用matplotlib显示图片
        # plt.imshow(img_rgb)
        # plt.show()
        # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # b, g, r = cv2.split(img)
        # b = b / 255.0
        # g = g / 255.0
        # r = r / 255.0
        # rgb_img = cv2.merge([r, g, b])
        img_rgb = img_rgb.reshape(-1, 3)
        img_rgb = np.expand_dims(img_rgb, axis=0)
        if rgb_image.size == 0:
            rgb_image = img_rgb
        else:
            rgb_image = np.vstack((rgb_image, img_rgb))

        # 读取深度图
        depthfile = os.path.join(depth_path, depth_files[i])
        depth = read_depth(depthfile)
        # depth = read_depth_from_txt(depthfile)
        # depth = np.loadtxt(depthfile)

        all_depths.append(depth)
        depth = depth[::sample_rate, ::sample_rate]
        depth = depth.flatten()
        if depth_map.size == 0:
            depth_map = depth
        else:
            depth_map = np.vstack((depth_map, depth))
    # 取特征点
    keypoints, matches = get_kpts_matches(img_path, keypoints_num, superglue)
    num = 0
    kpts_map = np.empty(0)
    kpts_rgb = np.empty(0)
    sample_points_num = depth_map.shape[1]
    all_matches = []
    for i in range(len(img_files)):
    # for i in range(2):
        kpt = keypoints[i]
        depth = all_depths[i]
        rgb = all_rgbs[i]
        kpt_indices = kpt[:, [1, 0]].astype(int)  # 调整索引顺序并转换为整数
        depth_values = depth[kpt_indices[:, 0], kpt_indices[:, 1]]
        rgb_values = rgb[kpt_indices[:, 0], kpt_indices[:, 1]]
        rgb_values =  np.expand_dims(rgb_values.reshape(-1, 3), axis=0)
        if kpts_map.size == 0:
            kpts_map = depth_values
        else:
            kpts_map = np.vstack((kpts_map, depth_values))
        if kpts_rgb.size == 0:
            kpts_rgb = rgb_values
        else:
            kpts_rgb = np.vstack((kpts_rgb, rgb_values))

        for j in range(i + 1, len(img_files)):
        # for j in range(i + 1, 2):
            match = matches[num]
            valid_match = match != -1
            valid_indice = np.where(valid_match)[0]
            valid_value = match[valid_match]
            valid_indice = valid_indice + (i + 1) * sample_points_num + i * keypoints_num
            valid_value = valid_value + (j + 1) * sample_points_num + j * keypoints_num
            # 构造索引矩阵
            index_matrix = np.column_stack((valid_indice, valid_value))
            all_matches.append(index_matrix)
            num += 1
    origins, directions, center_dir, all_origins, all_directions = get_ray(data_folder, sample_rate, np.array(keypoints))
    rgb_image = np.hstack((rgb_image, kpts_rgb))
    rgb_image = rgb_image.reshape(-1, 3)
    rays = {
        "origins": origins,
        "directions": directions,
        "colors": rgb_image,
    }
    depth_map = np.hstack((depth_map, kpts_map))
    cos = compute_cosine_similarity(directions.reshape(depth_map.shape[0], -1, 3), center_dir)
    depth_map = depth_map / cos
    all_matches = np.vstack(all_matches)
    all_points = get_points(np.array(all_depths), np.array(all_origins), np.array(all_directions), center_dir)
    return depth_map, rays, all_matches, all_points

def get_points(all_depth, all_origins, all_directions, center_dir):
    all_depth = all_depth.reshape(all_depth.shape[0], -1)
    all_directions = all_directions.reshape(all_directions.shape[0], -1, 3)
    all_origins = all_origins.reshape(all_origins.shape[0], -1, 3)
    cos = compute_cosine_similarity(all_directions, center_dir)
    all_depth = all_depth / cos
    all_points = all_depth[:, :, newaxis] * all_directions + all_origins
    return all_points


def compute_cosine_similarity(directions, center_directions):
    """
    计算方向张量与中心方向张量之间的余弦值。

    :param directions: 形状为 (32, 13288, 3) 的方向张量
    :param center_directions: 形状为 (32, 3) 的中心方向张量
    :return: 形状为 (32, 13288) 的余弦相似度矩阵
    """
    # 将中心方向张量扩展到与方向张量相同的形状
    expanded_center_directions = center_directions[:, np.newaxis, :]

    # 计算点积
    dot_product = np.sum(directions * expanded_center_directions, axis=-1)

    # 计算模长
    norm_directions = np.linalg.norm(directions, axis=-1)
    norm_center_directions = np.linalg.norm(expanded_center_directions, axis=-1)

    # 计算余弦相似度
    cosine_similarities = dot_product / (norm_directions * norm_center_directions)

    return cosine_similarities



def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo
}