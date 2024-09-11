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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, offset_sigmoid, scale_relu
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import torch.nn.functional as F
import open3d as o3d


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1),
                                        rotation).permute(0, 2, 1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:, :3, :3] = RS
            trans[:, 3, :3] = center
            trans[:, 3, 3] = 1
            return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        self.depth_scale_activation = scale_relu
        self.depth_offset_activation = offset_sigmoid

    def __init__(self, sh_degree: int, depth_map, rays, matches, all_points):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        # self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        # self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.depth_map_shape = depth_map.shape
        self.depth_map = depth_map.flatten().requires_grad_(False)  # 相对深度
        self.rays = rays  # 射线, origins, directions, colors
        self.voxel_size = 0.1

        self._depth_offset = torch.empty(0)  # 深度偏移
        self._depth_scale = torch.empty(0)  # 相对深度到绝对深度尺度

        self.depth_layers = torch.empty(0)

        self.matches = torch.tensor(matches).cuda()  # 匹配的点对，用以计算匹配的3d点距离loss
        # self.matches = None

        self.origin_points = torch.tensor(all_points) # 深度图直接转换成的点云，用以计算法线loss

        self.prune_mask = torch.ones(depth_map.shape[0] * depth_map.shape[1], dtype=torch.bool, device='cuda')

        self.select_view = -1

        self.isoffset_Save = False

    def match_dist_loss(self):
        points3d = self.get_xyz
        depth_scale = self.get_depth_scale
        # 提取匹配点
        point1 = points3d[self.matches[:, 0], :]
        point2 = points3d[self.matches[:, 1], :]

        pic_num1 = self.matches[:, 0] // self.depth_map_shape[1]
        pic_num2 = self.matches[:, 1] // self.depth_map_shape[1]
        # print(pic_num2)
        scale1 = depth_scale[pic_num1].squeeze()
        scale2 = depth_scale[pic_num2].squeeze()
        scale = scale1.mean() * scale2.mean()
        # 计算匹配点之间的距离
        distances = torch.norm(point1 - point2, dim=1)
        distances = distances / scale
        # 计算距离均值
        mean_distance = distances.mean()
        return mean_distance

    def visualize_pcd_with_lines(self, points, colors, point1, point2):
        # 创建点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.cpu().detach().numpy())
        pcd.colors = o3d.utility.Vector3dVector(colors)

        point1 = point1.cpu().detach().numpy()
        point2 = point2.cpu().detach().numpy()

        # 创建线段
        lines = []
        for i in range(point1.shape[0]):
            lines.append([i, i + point1.shape[0]])  # 假设 point1 和 point2 的长度相同

        # 创建 LineSet
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.vstack((point1, point2))),
            lines=o3d.utility.Vector2iVector(lines)
        )

        # 设置线的颜色
        line_set.paint_uniform_color([0, 0, 1])  # 设置线的颜色为蓝色

        # 创建一个可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # 添加点云和线段到可视化器
        vis.add_geometry(pcd)
        vis.add_geometry(line_set)

        # 设置点的大小
        opt = vis.get_render_option()
        opt.point_size = 10

        # 渲染并显示点云和线段
        vis.run()
        vis.destroy_window()


    def get_pcd_from_depth(self, isoffset=False):
        # depth_scale_layers = self.depth_layers * self.get_depth_scale[:, :, np.newaxis]
        depth_scale_expanded = self.get_depth_scale.expand(-1, self.depth_map_shape[1])
        depth_scale = depth_scale_expanded.flatten()[self.prune_mask]
        # depth_scale = depth_scale_expanded.flatten()[mask]
        if isoffset:
            depth_map_offset = torch.add(self.depth_map, self.get_depth_offset)
        else:
            depth_map_offset = self.depth_map

        AbsoluteDepth = depth_map_offset * depth_scale
        # AbsoluteDepth = torch.add(self.depth_map, self._depth_offset) * self._depth_scale[:, np.newaxis]
        # AbsoluteDepth = torch.sum(self.depth_map[:, np.newaxis, :] * depth_scale_layers, axis=1)
        # AbsoluteDepth = AbsoluteDepth.flatten()
        origins = torch.tensor(self.rays["origins"]).float().cuda()
        directions = torch.tensor(self.rays["directions"]).float().cuda()
        directions = F.normalize(directions, p=2, dim=1).float().cuda()
        points = torch.add(origins, AbsoluteDepth[:, np.newaxis] * directions).float()
        # points = torch.add((torch.tensor(self.rays["origins"]).cuda(), AbsoluteDepth[:, np.newaxis] * torch.tensor(self.rays["directions"]).cuda()).float())
        colors = self.rays["colors"]
        return points, colors

    def create_from_depth_pcd(self, spatial_lr_scale: float, percent_depth: float, depth_layers=1):
        print("create_from_depth_pcd")
        self.spatial_lr_scale = spatial_lr_scale

        # 深度分层
        # self.depth_layers = self.generate_layers(depth_layers)

        depth_offset = torch.zeros(self.depth_map_shape[0] * self.depth_map_shape[1], dtype=torch.float, device="cuda")
        # max = torch.max(self.depth_map, dim=1).values
        # min = torch.min(self.depth_map, dim=1).values
        # threshold = min + percent_depth * (max - min)
        # small_mask = self.depth_map < threshold[:, np.newaxis]
        # depth_offset[small_mask] = 1.0
        # depth_scale = torch.ones(self.depth_map.shape[0], self.depth_map.shape[1], dtype=torch.float, device="cuda") * 8
        depth_scale = torch.ones(self.depth_map_shape[0], depth_layers, dtype=torch.float, device="cuda")
        self._depth_offset = nn.Parameter(depth_offset.requires_grad_(True))
        self._depth_scale = nn.Parameter(depth_scale.requires_grad_(True))

        points, colors = self.get_pcd_from_depth()

        fused_point_cloud = points
        fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(points), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))

        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def generate_layers(self, num_layers):
        depth_layers = torch.zeros(self.depth_map_shape[0], num_layers, self.depth_map_shape[1], device="cuda")
        for i in range(self.depth_map_shape[0]):
            depth_map = self.depth_map[i, :]
            # 获取深度图的最小值和最大值
            min_depth = torch.min(depth_map)
            max_depth = torch.max(depth_map)

            # 计算每层的深度间隔
            depth_interval = (max_depth - min_depth + 1) / num_layers

            # 创建空列表来存储每一层的图像
            layers = []

            for j in range(num_layers):
                # 定义当前层的深度范围
                low = min_depth + j * depth_interval
                high = min_depth + (j + 1) * depth_interval
                # print("第{f}层的深度范围: {:.2f} - {:.2f}", i, low, high)

                # 创建当前层的掩膜
                mask = torch.logical_and(depth_map >= low, depth_map < high)

                # 应用掩膜得到当前层的图像
                layer_image = torch.zeros_like(depth_map)
                layer_image[mask] = 1  # 设置为1

                # 将当前层添加到列表中
                layers.append(layer_image)

            layers_tensor = torch.stack(layers)

            depth_layers[i, :, :] = layers_tensor

        return depth_layers

    def update_depth(self):
        self.depth_map = torch.add(self.depth_map, self.get_depth_offset)
        return


    def add_points(self, new_points, new_colors):
        self._xyz = torch.cat([self._xyz, new_points], dim=0)
        # self.max_radii2D = self.max_radii2D[valid_points_mask]

        fused_point_cloud = new_points
        fused_color = RGB2SH(new_colors)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at adding : ", fused_point_cloud.shape[0])

        # dist2 = torch.clamp_min(distCUDA2(self._xyz), 0.0000001)
        # scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        # self._scaling = nn.Parameter(scales.requires_grad_(True))
        # self._rotation = nn.Parameter(rots.requires_grad_(True))
        # self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        # new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        # new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        # new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        # new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        # new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_xyz = fused_point_cloud
        new_features_dc = features[:, :, 0:1].transpose(1, 2)
        new_features_rest = features[:, :, 1:].transpose(1, 2)
        new_opacities = opacities
        # new_scaling = scales
        new_scaling = torch.tensor(self.voxel_size).repeat(new_points.shape[0], 2).cuda()
        new_rotation = rots
        self.densification_postfix(new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation)

    def capture(self):
        return (
            self.active_sh_degree,
            # self._xyz,
            self._depth_offset,
            self._depth_scale,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            # self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         # self._xyz,
         self._depth_offset,
         self._depth_scale,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self.select_view != -1 and self.select_view < self.depth_map_shape[1]:
            length = self.depth_map_shape[1]
            return self.scaling_activation(self._scaling[self.select_view * length:(self.select_view + 1) * length])
        else:
            return self.scaling_activation(self._scaling)  # .clamp(max=1)
        # return self._scaling

    @property
    def get_rotation(self):
        if self.select_view != -1 and self.select_view < self.depth_map_shape[1]:
            length = self.depth_map_shape[1]
            return self.rotation_activation(self._rotation[self.select_view * length:(self.select_view + 1) * length])
        else:
            return self.rotation_activation(self._rotation)
        # return
    @property
    def get_xyz(self):
        if self.select_view != -1 and self.select_view < self.depth_map_shape[1]:
            points, _ = self.get_pcd_from_depth(True)
            length = self.depth_map_shape[1]
            return points[self.select_view * length:(self.select_view + 1) * length]
        elif self.isoffset_Save:
            points, _ = self.get_pcd_from_depth(True)
            return points
        else:
            points, _ = self.get_pcd_from_depth(True)
            return points

    # @property
    # def get_xyz(self):
    #     points, _ = self.get_pcd_from_depth()
    #     return points

    @property
    def get_features(self):
        if self.select_view != -1 and self.select_view < self.depth_map_shape[1]:
            length = self.depth_map_shape[1]
            features_dc = self._features_dc[self.select_view * length:(self.select_view + 1) * length]
            features_rest = self._features_rest[self.select_view * length:(self.select_view + 1) * length]
            return torch.cat((features_dc, features_rest), dim=1)
        else:
            features_dc = self._features_dc
            features_rest = self._features_rest
            return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        if self.select_view != -1 and self.select_view < self.depth_map_shape[1]:
            length = self.depth_map_shape[1]
            return self.opacity_activation(self._opacity[self.select_view * length:(self.select_view + 1) * length])
        else:
            return self.opacity_activation(self._opacity)

    @property
    def get_depth_offset(self):
        return self.depth_offset_activation(self._depth_offset)
        # return self._depth_offset

    @property
    def get_depth_scale(self):
        return self.depth_scale_activation(self._depth_scale)


    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
    #     self.spatial_lr_scale = spatial_lr_scale
    #     fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
    #     fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    #     features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
    #     features[:, :3, 0] = fused_color
    #     features[:, 3:, 1:] = 0.0
    #
    #     print("Number of points at initialisation : ", fused_point_cloud.shape[0])
    #
    #     dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
    #     scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
    #     rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
    #
    #     opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
    #
    #     self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    #     self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
    #     self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
    #     self._scaling = nn.Parameter(scales.requires_grad_(True))
    #     self._rotation = nn.Parameter(rots.requires_grad_(True))
    #     self._opacity = nn.Parameter(opacities.requires_grad_(True))
    #     self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        # self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            # {'params': [self._depth_offset], 'lr': training_args.depth_offset_lr_init * self.spatial_lr_scale, "name": "depth_offset"},
            {'params': [self._depth_scale], 'lr': training_args.depth_scale_lr_init * self.spatial_lr_scale,
             "name": "depth_scale"},
        ]

        self.optimizer = torch.optim.Adam(l, eps=1e-15)
        # self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
        #                                             lr_final=training_args.position_lr_final*self.spatial_lr_scale,
        #                                             lr_delay_mult=training_args.position_lr_delay_mult,
        #                                             max_steps=training_args.position_lr_max_steps)
        self.depth_scale_scheduler_args = get_expon_lr_func(
            lr_init=training_args.depth_scale_lr_init * self.spatial_lr_scale,
            lr_final=training_args.depth_scale_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.depth_scale_lr_delay_mult,
            max_steps=training_args.depth_scale_lr_max_steps)

        # self.depth_offset_scheduler_args = get_expon_lr_func(
        #     lr_init=training_args.depth_offset_lr_init * self.spatial_lr_scale,
        #     lr_final=training_args.depth_offset_lr_final * self.spatial_lr_scale,
        #     lr_delay_mult=training_args.depth_offset_lr_delay_mult,
        #     max_steps=training_args.depth_offset_lr_max_steps)


    def offset_training_setup(self, training_args):
        l = [
            {'params': [self._depth_offset], 'lr': training_args.depth_offset_lr_init * self.spatial_lr_scale,
             "name": "depth_offset"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, eps=1e-15)
        self.depth_offset_scheduler_args = get_expon_lr_func(
            lr_init=training_args.depth_offset_lr_init * self.spatial_lr_scale,
            lr_final=training_args.depth_offset_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.depth_offset_lr_delay_mult,
            max_steps=training_args.depth_offset_lr_max_steps)


    # def update_learning_rate(self, iteration):
    #     ''' Learning rate scheduling per step '''
    #     for param_group in self.optimizer.param_groups:
    #         if param_group["name"] == "xyz":
    #             lr = self.xyz_scheduler_args(iteration)
    #             param_group['lr'] = lr
    #             return lr

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "depth_scale":
                lr = self.depth_scale_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def update_learning_offset_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "depth_offset":
                lr = self.depth_offset_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def update_voxel_size(self, iteration):
        # if iteration > 1000:
        #     self.voxel_size = 0.01
        if iteration > 2000:
            self.voxel_size = 0.001
        return self.voxel_size

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        depth_scale = self.get_depth_scale.detach().cpu().numpy()
        depth_offset = self.get_depth_offset.detach().cpu().numpy()
        self.write_to_txt(depth_scale, depth_offset, path + 'scale&offset.txt')
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def write_to_txt(self, depth_scale, depth_offset, path):
        # 将数据写入到文本文件中
        with open(path, 'w') as file:
            # 写入 depth_scale
            file.write("Depth Scale:\n")
            np.savetxt(file, depth_scale, fmt='%.6f')

            # 写入 depth_offset
            file.write("\nDepth Offset:\n")
            np.savetxt(file, depth_offset, fmt='%.6f')

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_depth_offset(self):
        depth_offset_new = torch.zeros_like(self.get_depth_offset)
        optimizable_tensors = self.replace_tensor_to_optimizer(depth_offset_new, "depth_offset")
        self._depth_offset = optimizable_tensors["depth_offset"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] == 'depth_scale':
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        # print(valid_points_mask.)
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._depth_offset = optimizable_tensors["depth_offset"]
        # self.depth_scale = optimizable_tensors["depth_scale"]

        # self._xyz = self._xyz[valid_points_mask]
        # self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        # self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.depth_map = self.depth_map[valid_points_mask]
        self.rays['directions'] = self.rays['directions'][valid_points_mask.cpu()]
        self.rays['origins'] = self.rays['origins'][valid_points_mask.cpu()]
        self.rays['colors'] = self.rays['colors'][valid_points_mask.cpu()]

        # 更新匹配matches
        # 获取保留的点的索引
        valid_indices = torch.arange(len(mask), device='cuda')[~mask]

        # 创建索引映射表，将旧的索引映射到新的索引
        index_mapping = torch.full((len(mask),), -1, dtype=torch.long, device='cuda')
        index_mapping[valid_indices] = torch.arange(len(valid_indices), device='cuda')

        # 更新 matches 张量
        updated_matches = index_mapping[self.matches]

        # 保留有效的匹配对，忽略那些映射到 -1 的条目
        valid_matches_mask = (updated_matches >= 0).all(dim=1)
        valid_matches = updated_matches[valid_matches_mask]
        self.matches = valid_matches

        # 更新prune_mask
        # 计算 prune_mask 中为 True 的索引
        true_indices = torch.nonzero(self.prune_mask).squeeze()  # 这些索引是当前保留的点

        # 更新 prune_mask
        updated_prune_mask = self.prune_mask.clone()
        updated_prune_mask[true_indices[mask]] = False
        self.prune_mask = updated_prune_mask

    def prune(self, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "depth_scale" or group["name"] == "depth_offset":
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        # "xyz": new_xyz,
        d = {
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation}
        # self._xyz = optimizable_tensors["xyz"]
        # self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1