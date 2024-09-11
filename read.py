import numpy as np
from open3d import *
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def read_ply(path):
    # 读取PLY文件
    ply_data = PlyData.read(path)

    # 提取 'vertex' 元素
    vertex_data = ply_data['vertex']

    # 读取属性列表
    attributes = {attr: np.array(vertex_data[attr]) for attr in vertex_data.data.dtype.names}

    # 从属性字典中提取各个属性
    xyz = attributes['x'], attributes['y'], attributes['z']
    normals = attributes.get('nx', np.zeros_like(xyz[0]))  # 如果法线属性不存在，设置为零

    # 处理 'f_dc' 开头的属性
    f_dc_keys = [key for key in attributes.keys() if key.startswith('f_dc_')]
    if f_dc_keys:
        f_dc = np.stack([attributes[key] for key in f_dc_keys], axis=-1)
    else:
        f_dc = np.zeros((len(xyz[0]), 0))  # 如果没有 f_dc 属性，设置为空数组

    # 处理 'f_rest' 开头的属性
    f_rest_keys = [key for key in attributes.keys() if key.startswith('f_rest_')]
    if f_rest_keys:
        f_rest = np.stack([attributes[key] for key in f_rest_keys], axis=-1)
    else:
        f_rest = np.zeros((len(xyz[0]), 0))  # 如果没有 f_rest 属性，设置为空数组

    # 处理其他属性
    opacities = attributes.get('opacity', np.zeros(len(xyz[0])))
    scale = attributes.get('scale', np.zeros(len(xyz[0])))
    rotation = attributes.get('rotation', np.zeros(len(xyz[0])))
    opacities = sigmoid(opacities)
    mask = opacities >= 0.00
    # 合并坐标
    xyz = np.stack(xyz, axis=-1)
    color = SH2RGB(f_dc)
    normals = np.stack(normals, axis=-1)
    xyz = xyz[mask, :]
    color = color[mask, :]
    normals = normals[mask]
    opacities = opacities[mask]

    return xyz, color, normals




def visualization(patch, isread=True, iswrite=False):
    print("visualization start work")
    # deep = self.deep()
    if isread:
        points, colors, _ = read_ply(path)
    if points is None:
        print("empty")
    # print(point)
    points_array = np.array(points)
    colors_array = np.array(colors)
    # normals_array = np.array(normals)
    # print(points_array)
    print("points", points_array.shape)
    print("colors", colors_array.shape)

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points_array)
    if colors_array.size != 0:
        pcd.colors = open3d.utility.Vector3dVector(colors_array)
    pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    if iswrite:
        open3d.io.write_point_cloud("/home/liu/2d-gaussian-depth/data/table/output/ca5bf16c-b/point_cloud/iteration_2000/point_cloud.ply", pcd)

    # 创建 Visualizer 和 ViewControl
    vis = open3d.visualization.Visualizer()
    vis.create_window(visible=True)

    # 添加点云到场景中
    vis.add_geometry(pcd)

    # 创建坐标系
    coord_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1)

    # 添加三维坐标系
    # vis.add_geometry(open3d.geometry.TriangleMesh.create_coordinate_frame(size=20))
    vis.add_geometry(coord_frame)

    # 设置渲染参数并显示点云
    opt = vis.get_render_option()
    opt.point_size = 10  # 点的大小
    opt.background_color = np.asarray([0, 0, 0])  # 背景颜色

    vis.run()
    vis.destroy_window()
path = '/home/liu/2d-gaussian-depth/data/table/output/best/point_cloud/iteration_5000/point_cloud.ply'
visualization(path)

# read_ply('/home/liu/2d-gaussian-depth/output/0aff0500-c/point_cloud/iteration_30000/point_cloud.ply')