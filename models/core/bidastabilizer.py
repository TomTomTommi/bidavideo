from typing import Dict, List, ClassVar
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

import importlib
import sys

from bidavideo.models.core.update import (
    MultiSequenceUpdateBlock3D,
)
from bidavideo.models.core.extractor import BasicEncoder, ResidualBlock
from bidavideo.models.core.corr import TFCL

from bidavideo.models.core.utils.utils import InputPadder, interp
from bidavideo.models.sea_raft_model import SEARAFTModel

autocast = torch.cuda.amp.autocast


def make_layer(block, num_blocks, **kwarg):
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)

class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out

class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()
        main = []
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)

class BiDAStabilizer(nn.Module):
    def __init__(self, mid_channels=48, num_blocks=5):
        super(BiDAStabilizer, self).__init__()

        self.raft = SEARAFTModel()
        self.mid_channels = mid_channels

        self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)

        self.backward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + mid_channels, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + mid_channels, mid_channels, num_blocks)

        # upsample
        self.fusion = nn.Conv2d(
            mid_channels + mid_channels, mid_channels, 1, 1, 0, bias=True)
        self.conv_hr = nn.Conv2d(mid_channels, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 1, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def compute_flow(self, seq):
        n, t, c, h, w = seq.size()

        flows_forward_list = []
        flows_backward_list = []
        for i in range(t-1):
            # i-th flow_backward denotes seq[i+1] towards seq[i]
            flow_backward = self.raft.forward_fullres(seq[:,i], seq[:,i+1])
            # i-th flow_forward denotes seq[i] towards seq[i+1]
            flow_forward = self.raft.forward_fullres(seq[:,i+1], seq[:,i])
            flows_backward_list.append(flow_backward)
            flows_forward_list.append(flow_forward)
        flow_forward = torch.stack(flows_forward_list, dim=1)
        flow_backward = torch.stack(flows_backward_list, dim=1)

        return flow_forward, flow_backward

    def flow_warp(self, x, flow):
        if flow.size(3) != 2:  # [B, H, W, 2]
            flow = flow.permute(0, 2, 3, 1)
        if x.size()[-2:] != flow.size()[1:3]:
            raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                             f'flow ({flow.size()[1:3]}) are not the same.')
        _, _, h, w = x.size()
        # create mesh grid
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (h, w, 2)
        grid.requires_grad = False

        grid_flow = grid + flow
        # scale grid_flow to [-1,1]
        grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
        output = F.grid_sample(
            x,
            grid_flow,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True)
        return output

    def forward(self, seq1, disp):
        b, t, c, h, w = seq1.shape

        # compute optical flow
        flow_forward, flow_backward = self.compute_flow(seq1)

        disp_abs = -disp

        disp_backward_list = []
        for i in range(1, t):
            feat_prop = self.flow_warp(disp_abs[:, i], flow_backward[:, i - 1])
            disp_backward_list.append(feat_prop)
        disp_backward_list.append(disp_abs[:, t - 1])
        disp_backward = torch.stack(disp_backward_list, dim=1)

        output_forward_list = [disp_abs[:, 0]]
        for i in range(t - 1):
            feat_prop = self.flow_warp(disp_abs[:, i], flow_forward[:, i])
            output_forward_list.append(feat_prop)
        disp_forward = torch.stack(output_forward_list, dim=1)

        disp_concate = torch.cat([disp_forward, disp_abs, disp_backward], dim=2)
        feats_ = self.feat_extract(disp_concate.contiguous().view(-1, 3, h, w))
        feats_ = feats_.view(b, t, -1, h, w)

        # backward-time propgation
        outputs = []
        feat_prop = feats_.new_zeros(b, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                flow = flow_backward[:, i, :, :, :]
                feat_prop = self.flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([feats_[:, i, :, :, :], feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)

            outputs.append(feat_prop)
        outputs = outputs[::-1]

        # forward-time propagation
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            if i > 0:  # no warping required for the first timestep
                flow = flow_forward[:, i - 1, :, :, :]
                feat_prop = self.flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([feats_[:, i, :, :, :], feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)

            # refining given the backward and forward features
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = disp_abs[:, i, :, :, :]
            out += base
            outputs[i] = -out

        return torch.stack(outputs, dim=0)

    def forward_batch(self, video, disp, kernel_size=50):
        disp_preds_list = []
        num_ims = len(video)
        print("video", video.shape)
        if kernel_size >= num_ims:
            padder = InputPadder(video.shape, divis_by=32)
            video, disp = padder.pad(video, disp)

            disp_preds = self.forward(video[None], disp[None])
            disp_preds = padder.unpad(disp_preds.squeeze(1))

            return disp_preds[:,None]

        else:
            stride = kernel_size // 2
            for i in range(0, num_ims, stride):
                if min(i + kernel_size, num_ims) - i < 3:
                    disp_preds = disp[i: min(i + kernel_size, num_ims)]
                    disp_preds_list.append(disp_preds)
                    continue
                video_clip = video[i : min(i + kernel_size, num_ims)]
                disp_clip = disp[i: min(i + kernel_size, num_ims)]
                padder = InputPadder(video_clip.shape, divis_by=32)

                video_clip, disp_clip = padder.pad(video_clip, disp_clip)
                disp_preds = self.forward(video_clip[None], disp_clip[None])

                disp_preds = padder.unpad(disp_preds.squeeze(1))

                if len(disp_preds_list) > 0 and len(disp_preds) >= stride:

                    if len(disp_preds) < kernel_size:
                        disp_preds_list.append(disp_preds[stride // 2 :])
                        break
                    else:
                        disp_preds_list.append(disp_preds[stride // 2 : -stride // 2])

                elif len(disp_preds_list) == 0:
                    disp_preds_list.append(disp_preds[: -stride // 2])

            disp_preds = torch.cat(disp_preds_list, dim=0)

            return disp_preds[:,None]
