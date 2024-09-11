import argparse
import torch
import numpy as np

from SuperGlue.models.matching import Matching
from SuperGlue.models.utils import (VideoStreamer, frame2tensor)


# torch.set_grad_enabled(False)



def get_kpts_matches(input, max_keypoints, superglue):
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default=input,
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default='./output',
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[-1, -1],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default=superglue,
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=max_keypoints,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.0005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.8,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true', default=True,
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true', default=False,
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors', 'image']

    vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)
    all_data =[]
    all_keypoints = []
    all_matches = []

    while True:
        frame, ret = vs.next_frame()
        if not ret:
            print('Finished demo_superglue.py')
            break
        frame_tensor = frame2tensor(frame, device)
        last_data = matching.superpoint({'image': frame_tensor})
        # last_data = {k+'0': last_data[k] for k in keys}
        last_data['image'] = frame_tensor
        keypoints_numpy = last_data['keypoints'][0].cpu().numpy()
        all_keypoints.append(keypoints_numpy)
        all_data.append(last_data)
    vs.cleanup()
    for i in range(len(all_data)):
        for j in range(i + 1, len(all_data)):
            data0 = {k + '0': all_data[i][k] for k in keys}
            data1 = {k + '1': all_data[j][k] for k in keys}
            data = {**data0, **data1}
            pred = matching(data)
            matches = pred['matches0'][0].cpu().numpy()
            all_matches.append(matches)
    # 清理内存
    del matching
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)
    return all_keypoints, all_matches



#
# if __name__ == '__main__':
#     keypoints, matches = get_kpts_matches('/home/liu/2d-gaussian-depth/data/table/images', 200, 'indoor')
#     print("keypoints", keypoints)
#     print("matches", matches)
