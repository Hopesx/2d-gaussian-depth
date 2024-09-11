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
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.dataset_readers import get_depths_rays
from PIL import Image
from utils.camera_utils import check_camera_pose
import time

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    depth_map, rays, matches, all_points = get_depths_rays(dataset.source_path, 12, 2000,'outdoor')
    # gaussians = GaussianModel(dataset.sh_degree)
    gaussians = GaussianModel(dataset.sh_degree, torch.from_numpy(depth_map).cuda(), rays, matches, all_points)
    # gaussians.init_depth(depth_map, rays)


    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    scene.save(0)

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        gaussians.update_learning_offset_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        random_cam = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(random_cam)
        # view_gaussians = gaussians.pop_gs(viewpoint_cam)

        if iteration == opt.depth_offset_until_iter:
            gaussians.select_view = -1
            gaussians.update_depth()
            gaussians.training_setup(opt)

        if iteration == opt.depth_offset_from_iter:
            gaussians.offset_training_setup(opt)

        if iteration > opt.depth_offset_from_iter and iteration <= opt.depth_offset_until_iter:
            # 优化深度图，找到若干和viewpoint_cam相近的视角，优化这些视角的深度图
            test_viewpoint_cameras = scene.getTrainCameras().copy()
            all_loss = []
            num = 0
            # loss = 0

            while True:
                random_test_cam = randint(0, len(test_viewpoint_cameras) - 1)
                test_camera = test_viewpoint_cameras.pop(random_test_cam)
                if check_camera_pose(viewpoint_cam, test_camera):
                    gaussians.select_view = test_camera.uid
                    render_pkg = render(viewpoint_cam, gaussians, pipe, background, isoffset=True)
                    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    gt_image = test_camera.original_image.cuda()
                    bg_color_tensor = background.view(3, 1, 1)
                    mask = torch.all(image == bg_color_tensor, dim=0)
                    mask_expanded = ~ mask.unsqueeze(0)
                    mask_expanded = mask_expanded.expand_as(image)
                    # masked_image = image * mask_expanded
                    # masked_gt = gt_image * mask_expanded
                    Ll1 = torch.abs((image - gt_image))[mask_expanded].mean()
                    single_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                    all_loss.append(single_loss)
                    num += 1
                if num > 2:
                    break
                if len(test_viewpoint_cameras) == 0:
                    break
            loss = torch.stack(all_loss).mean()
            total_loss = loss

            total_loss.backward()

            iter_end.record()

            gaussians.select_view = -1

            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    loss_dict = {
                        "total_loss": f"{total_loss.item():.{5}f}",
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "Points": f"{len(gaussians.get_xyz)}"
                    }
                    progress_bar.set_postfix(loss_dict)

                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            gt_image = viewpoint_cam.original_image.cuda()

            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
            # regularization
            lambda_normal = opt.lambda_normal if iteration > 1000 else 0.0
            lambda_dist = opt.lambda_dist if iteration > 1000 else 0.0

            rend_dist = render_pkg["rend_dist"]
            rend_normal = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            depth_normal = render_pkg['depth_normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            depth_normal_error = (1 - (depth_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()
            depth_normal_loss = lambda_normal * (depth_normal_error).mean()

            # loss
            # total_loss = loss + dist_loss + normal_loss

            match_loss = gaussians.match_dist_loss() * 10

            total_loss = loss + match_loss + depth_normal_loss + normal_loss

            total_loss.backward()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
                ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

                if iteration % 10 == 0:
                    loss_dict = {
                        "total_loss": f"{total_loss.item():.{5}f}",
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "match_loss": f"{match_loss.item():.{5}f}",
                        # "scale": f"{gaussians.get_depth_scale[0, :].mean().item(), gaussians.get_depth_scale[1, :].mean().item()}",
                        "distort": f"{ema_dist_for_log:.{5}f}",
                        # "normal": f"{ema_normal_for_log:.{5}f}",
                        "normal": f"{depth_normal_loss.item():.{5}f}",
                        "Points": f"{len(gaussians.get_xyz)}"
                    }
                    progress_bar.set_postfix(loss_dict)

                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                if tb_writer is not None:
                    tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                    tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                testing_iterations, scene, render, (pipe, background))
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.gaussians.isoffset_Save = True
                    scene.save(iteration)
                    scene.gaussians.isoffset_Save = False

                # Prune
                if iteration < opt.prune_until_iter and match_loss < 0.05:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])
                    if iteration > opt.prune_from_iter and iteration % opt.prune_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.prune(opt.opacity_cull, scene.cameras_extent, size_threshold)


        # print("offset_grad", gaussians.get_depth_offset.grad)
        # print("scale_grad", gaussians.get_depth_scale.grad)


        # gaussians.updateGS()
        # print("offset_grad", gaussians.get_depth_offset.grad)
        # print("scale_grad", gaussians.get_depth_scale.grad)
        # print("offset", gaussians.get_depth_offset)
        # print("scale", gaussians.get_depth_scale)
        # print("xyz", gaussians.get_xyz)



        with torch.no_grad():

            # Densification
            # if iteration < opt.depth_offset_reset_until_iter:
            #     if iteration % opt.depth_offset_reset_interval == 0:
            #         gaussians.reset_depth_offset()

                # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 从深度信息中更新高斯
                # pcd = gaussians.get_pcd_from_depth()
                # gaussians.updateGS()
                # 更新体素网格大小
                # gaussians.update_voxel_size(iteration)

                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #     gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)

                # reset opacity and depth offset
                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                # if iteration > opt.depth_offset_from_iter and iteration < opt.depth_offset_until_iter:
                #     print(gaussians.select_view)
                #     print(gaussians.get_depth_offset[gaussians.select_view])

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

            if iteration % 1000 == 0:
                # 测试渲染结果
                l1_test = 0.0
                psnr_test = 0.0
                viewpoint_stack_test = scene.getTrainCameras().copy()
                a = len(viewpoint_stack_test)
                for i in range(len(viewpoint_stack_test)):
                    viewpoint_cam_test = viewpoint_stack_test[i]
                    # view_gaussians = gaussians.pop_gs(viewpoint_cam)
    
                    render_pkg_test = render(viewpoint_cam_test, gaussians, pipe, background)
                    image_test = render_pkg_test["render"]
                    gt_image = viewpoint_cam.original_image.cuda()

                    l1_test += l1_loss(image_test, gt_image).mean().double()
                    psnr_test += psnr(image_test, gt_image).mean().double()
    
                    # 将 tensor 转换成 numpy 数组
                    image_num = image_test.clone().cpu().detach().numpy()
    
                    # 如果 tensor 是 (C, H, W) 格式，转换成 (H, W, C)
                    if image_num.shape[0] == 3:
                        image_num = np.transpose(image_num, (1, 2, 0))
    
                    # 将数据类型转换为 uint8
                    image_num = (image_num * 255).astype(np.uint8)
    
                    # 创建 PIL Image 对象
                    pil_image = Image.fromarray(image_num)
    
                    if not os.path.exists(f"{scene.model_path}/test/{iteration}"):
                        os.makedirs(f"{scene.model_path}/test/{iteration}")
    
                    # 保存图像
                    pil_image.save(f"{scene.model_path}/test/{iteration}/{viewpoint_cam_test.uid}.png")
                    # image.savefig("output/%d.png" % iteration, bbox_inches='tight')
                psnr_test /= len(viewpoint_stack_test)
                l1_test /= len(viewpoint_stack_test)
                with open(f"{scene.model_path}/test/{iteration}/metrics.txt", 'w') as f:
                    f.write(f"L1 Loss: {l1_test}\n")
                    f.write(f"PSNR: {psnr_test}\n")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    else:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join(args.model_path, unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        # depth = render_pkg["surf_depth"]
                        # norm = depth.max()
                        # depth = depth / norm
                        # depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        # tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_00, 20_00, 50_00, 100_00])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_00, 20_00, 30_00, 50_00, 100_00, 200_00])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")