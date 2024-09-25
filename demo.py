import sys

import argparse
import os
import cv2
import glob
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path

DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile).convert('RGB')).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img.to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey()


def demo(args):
    if args.model_name == "bidastereo":
        from bidavideo.models.bidastereo_model import BiDAStereoModel
        model = BiDAStereoModel()
    elif args.model_name =="raftstereo":
        from bidavideo.models.raft_stereo_model import RAFTStereoModel
        model = RAFTStereoModel()
    elif args.model_name =="igevstereo":
        from bidavideo.models.igev_stereo_model import IGEVStereoModel
        model = IGEVStereoModel()
    else:
        raise ValueError("Wrong model name:", args.model_name)

    if args.ckpt is not None:
        assert args.ckpt.endswith(".pth") or args.ckpt.endswith(
            ".pt"
        )
        strict = True
        state_dict = torch.load(args.ckpt)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
        model.model.load_state_dict(state_dict, strict=strict)

        print("Done loading model checkpoint", args.ckpt)

    if args.stabilizer:
        from bidavideo.models.core.bidastabilizer import BiDAStabilizer
        model_stabilizer = BiDAStabilizer()
        model_stabilizer.cuda()
        if args.stabilizer_ckpt is not None:
            assert args.stabilizer_ckpt.endswith(".pth") or args.stabilizer_ckpt.endswith(".pt")
            state_dict = torch.load(args.stabilizer_ckpt)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
            model_stabilizer.load_state_dict(state_dict, strict=True)
            print("Done loading stabilizer checkpoint:", args.stabilizer_ckpt)

    model.to(DEVICE)
    model.eval()

    output_directory = args.output_path
    parent_directory = os.path.dirname(output_directory)
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    with torch.no_grad():
        images_left = sorted(glob.glob(os.path.join(args.path, 'left/*.png')) + glob.glob(os.path.join(args.path, 'left/*.jpg')))
        images_right = sorted(glob.glob(os.path.join(args.path, 'right/*.png')) + glob.glob(os.path.join(args.path, 'right/*.jpg')))
        assert len(images_left) == len(images_right), [len(images_left), len(images_right)]
        assert len(images_left) > 0, args.path
        print(f"Found {len(images_left)} frames. Saving files to {args.output_path}")

        num_frames = len(images_left)
        frame_size = args.frame_size

        disparities_ori_all = []

        for start_idx in range(0, num_frames, frame_size):
            end_idx = min(start_idx + frame_size, num_frames)

            image_left_list = []
            image_right_list = []

            for imfile1, imfile2 in zip(images_left[start_idx:end_idx], images_right[start_idx:end_idx]):
                image_left = load_image(imfile1)
                image_right = load_image(imfile2)
                image_left = F.interpolate(image_left[None], size=args.resize, mode="bilinear", align_corners=True)
                image_right = F.interpolate(image_right[None], size=args.resize, mode="bilinear", align_corners=True)
                image_left_list.append(image_left[0])
                image_right_list.append(image_right[0])

            video_left = torch.stack(image_left_list, dim=0)
            video_right = torch.stack(image_right_list, dim=0)

            batch_dict = defaultdict(list)
            batch_dict["stereo_video"] = torch.stack([video_left, video_right], dim=1)

            if args.stabilizer:
                predictions = model.forward_stabilizer(batch_dict, model_stabilizer)
            else:
                predictions = model(batch_dict)

            assert "disparity" in predictions
            disparities = predictions["disparity"][:, :1].clone().data.cpu().abs().numpy()
            disparities_ori = disparities.astype(np.uint8)
            disparities_ori_all.extend(disparities_ori)

        disparities_ori_all = np.array(disparities_ori_all)
        disparities_all = ((disparities_ori_all - disparities_ori_all.min()) / (disparities_ori_all.max() - disparities_ori_all.min()) * 255).astype(np.uint8)

        video_ori_disparity = cv2.VideoWriter(
            os.path.join(args.output_path, "disparity.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps=args.fps,
            frameSize=(disparities_all.shape[3], disparities_all.shape[2]),
            isColor=True,
        )
        video_disparity = cv2.VideoWriter(
            os.path.join(args.output_path, "disparity_norm.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps=args.fps,
            frameSize=(disparities_all.shape[3], disparities_all.shape[2]),
            isColor=True,
        )

        for i in range(num_frames):
            imfile1 = images_left[i]
            file_stem = imfile1.split('/')[-1].split('-')[0]

            disparity_norm = disparities_all[i]
            disparity_norm = disparity_norm.transpose(1, 2, 0)
            disparity_norm_vis = cv2.applyColorMap(disparity_norm, cv2.COLORMAP_INFERNO)
            video_disparity.write(disparity_norm_vis)

            disparity_ori = disparities_ori_all[i]
            disparity_ori = disparity_ori.transpose(1, 2, 0)
            disparity_ori_vis = cv2.applyColorMap(disparity_ori, cv2.COLORMAP_INFERNO)
            video_ori_disparity.write(disparity_ori_vis)

            if args.save_png:
                filename_temp = args.output_path + '/disparity_norm_' + str(i).zfill(3) + '.png'
                cv2.imwrite(filename_temp, disparity_norm_vis)
                filename_temp = args.output_path + '/disparity_ori_' + str(i).zfill(3) + '.png'
                cv2.imwrite(filename_temp, disparity_ori_vis)

        video_ori_disparity.release()
        video_disparity.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="bidastereo", help="name to specify model")
    parser.add_argument('--ckpt', default=None, help="checkpoint of stereo model")
    parser.add_argument("--stabilizer", action="store_true")
    parser.add_argument('--stabilizer_ckpt', default=None, help="checkpoint of stabilizer model")
    parser.add_argument('--resize', default=(720, 1280), help="image size input to the model")
    parser.add_argument("--fps", type=int, default=10, help="frame rate for video visualization")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument("--save_png", action="store_true")
    parser.add_argument("--frame_size", type=int, default=150, help="number of updates in each forward pass.")
    parser.add_argument("--iters",type=int, default=20, help="number of updates in each forward pass.")
    parser.add_argument("--kernel_size", type=int, default=20, help="number of frames in each forward pass.")
    parser.add_argument('--output_path', help="directory to save output", default="demo_output")
    args = parser.parse_args()

    demo(args)
