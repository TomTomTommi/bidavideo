# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from typing import ClassVar
import torch.nn.functional as F

from pytorch3d.implicitron.tools.config import Configurable
import torch
import importlib
import sys
import os

autocast = torch.cuda.amp.autocast

class SEARAFTModel(Configurable, torch.nn.Module):
    MODEL_CONFIG_NAME: ClassVar[str] = "SEARAFTModel"

    def __post_init__(self):
        # sys.path.append("third_party/SEA-RAFT")
        thirdparty_searaft_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../third_party/SEA-RAFT"))
        sys.path.append(thirdparty_searaft_path)
        sea_raft = importlib.import_module(
            "bidavideo.third_party.SEA-RAFT.core.raft"
        )
        self.raft_utils = importlib.import_module(
            "bidavideo.third_party.SEA-RAFT.core.utils.utils"
        )
        super().__init__()
        self.model_weights: str = "./third_party/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-S.pth"

        model_args = SimpleNamespace(
            use_var= True,
            var_min= 0,
            var_max= 10,
            pretrain= "resnet18",
            initial_dim= 64,
            block_dims= [64, 128, 256],
            radius= 4,
            dim= 128,
            num_blocks= 2,
            iters= 4
        )
        self.args = model_args
        self.model = sea_raft.RAFT(model_args).cuda()

        state_dict = torch.load(self.model_weights, map_location="cpu")
        weight_dict = {}
        for k,v in state_dict.items():
            temp_k = k.replace('module.', '') if 'module' in k else k
            weight_dict[temp_k] = v
        self.model.load_state_dict(weight_dict, strict=True)


    def forward(self, image1, image2):
        left_image_rgb = image1.cuda()
        right_image_rgb = image2.cuda()
        padder = self.raft_utils.InputPadder(left_image_rgb.shape)
        left_image_rgb, right_image_rgb = padder.pad(
            left_image_rgb, right_image_rgb
        )
        output = self.model(left_image_rgb, right_image_rgb, iters=4, test_mode=True)
        flow_up = output['flow'][-1]
        flow_up = padder.unpad(flow_up)
        return 0.25 * F.interpolate(flow_up, size=(flow_up.shape[2] // 4, flow_up.shape[3] // 4), mode="bilinear",
        align_corners=True)

    def forward_fullres(self, image1, image2):
        left_image_rgb = image1.cuda()
        right_image_rgb = image2.cuda()
        padder = self.raft_utils.InputPadder(left_image_rgb.shape)
        left_image_rgb, right_image_rgb = padder.pad(
            left_image_rgb, right_image_rgb
        )
        output = self.model(left_image_rgb, right_image_rgb, iters=4, test_mode=True)
        flow_up = output['flow'][-1]
        flow_up = padder.unpad(flow_up)
        return flow_up
