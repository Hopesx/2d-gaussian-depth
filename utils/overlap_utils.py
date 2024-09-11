import torch
# import faiss

def compute_overlap_rate(point_clouds, threshold, batch_size):
    """Compute the overlap rates between point clouds"""
    n, num_points, _ = point_clouds.shape
    overlap_counts = torch.zeros((n, n), device=point_clouds.device)

    # Compute distances using cdist
    for i in range(n):
        for j in range(i + 1, n, batch_size):
            batch_j_end = min(j + batch_size, n)
            pc1 = point_clouds[i].unsqueeze(0)  # Shape: (1, num_points, 3)
            pc2 = point_clouds[j:batch_j_end]
            # Compute pairwise distances for the batch
            distances = torch.cdist(pc1, pc2)  # Shape: (1, n, num_points)
            count = torch.sum(distances < threshold).float()
            overlap_counts[i, j] = count
    overlap_rates = torch.sum(overlap_counts) / (n - 1) / (n - 2)
    return overlap_rates


def smooth_unique(x, threshold=0.1):
    """
    近似的去重方法，通过将数据映射到一个平滑的空间来保持可微性。

    Args:
        x (torch.Tensor): 输入张量。
        threshold (float): 去重的阈值。

    Returns:
        torch.Tensor: 近似的去重结果。
    """
    x = x / threshold
    x = torch.round(x) * threshold
    return torch.unique(x, dim=0)

def activate_grids(point_clouds, grid_size, threshold):
    all_grids = []
    n, num_points, _ = point_clouds.shape
    for i in range(n):
        pc = point_clouds[i]
        # 计算网格坐标
        grid_coords = (pc / grid_size).floor().long()  # 使用 floor 而不是 //，保持可微性
        distance = torch.cdist(grid_coords.unsqueeze(0), grid_coords.unsqueeze(0))


        # 批量处理，避免逐点的集合操作
        grids.append(grid_coords)

    # 将所有点云的网格坐标合并
    all_grids = torch.cat(all_grids, dim=0)
    unique_points = smooth_unique(points, threshold=0.1)

# def compute_overlap_rate_faiss(point_clouds, threshold):
#     """Compute the overlap rates using Faiss for distance calculation"""
#     device = point_clouds.device
#     n, num_points, _ = point_clouds.shape
#
#     overlap_counts = torch.zeros((n, n), device=device)
#
#     # Flatten point clouds
#     pc_flat = point_clouds.view(n * num_points, 3).cpu().numpy()
#
#     # Create a Faiss index
#     index = faiss.IndexFlatL2(3)
#
#     # Add point clouds to index
#     index.add(pc_flat)
#
#     for i in range(n):
#         # Compute distances for point cloud i
#         start_idx = i * num_points
#         end_idx = (i + 1) * num_points
#         query_points = pc_flat[start_idx:end_idx]
#
#         # Search in index
#         D, I = index.search(query_points, num_points)  # D: distances, I: indices
#
#         # Count overlaps
#         for j in range(n):
#             if i != j:
#                 start_idx_j = j * num_points
#                 end_idx_j = (j + 1) * num_points
#                 distances = D[start_idx:end_idx]
#                 overlaps = (distances < threshold).sum()
#                 overlap_counts[i, j] = overlaps
#
#     overlap_counts = overlap_counts.triu(diagonal=1)
#     num_comparisons = (n - 1) * (n - 2) / 2
#     overlap_rate = torch.sum(overlap_counts) / num_comparisons
#
#     print(overlap_rate)
#     return overlap_rate

