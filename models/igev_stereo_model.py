# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from types import SimpleNamespace
from typing import ClassVar

import torch
from pytorch3d.implicitron.tools.config import Configurable

import importlib
import sys
import os

autocast = torch.cuda.amp.autocast


class IGEVStereoModel(Configurable, torch.nn.Module):

    MODEL_CONFIG_NAME: ClassVar[str] = "IGEVStereoModel"
    model_weights: str = "./checkpoints/igevstereo_robust/igevstereo_robust.pth"
    kernel_size: int = 50

    def __post_init__(self):
        super().__init__()
        # sys.path.append("third_party/IGEV-Stereo")
        thirdparty_igev_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../third_party/IGEV-Stereo"))
        sys.path.append(thirdparty_igev_path)
        igev_stereo = importlib.import_module(
            "bidavideo.third_party.IGEV-Stereo.core.igev_stereo"
        )
        self.igev_stereo_utils = importlib.import_module(
            "bidavideo.third_party.IGEV-Stereo.core.utils.utils"
        )

        model_args = SimpleNamespace(
            hidden_dims=[128] * 3,
            corr_implementation="reg",
            shared_backbone=False,
            corr_levels=2,
            corr_radius=4,
            n_downsample=2,
            slow_fast_gru=False,
            n_gru_layers=3,
            mixed_precision=False,
            max_disp=512,
        )
        self.args = model_args
        model =  igev_stereo.IGEVStereo(model_args).cuda()
        state_dict = torch.load(self.model_weights, map_location="cpu")

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            state_dict = {"module." + k: v for k, v in state_dict.items()}
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        else:
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=True)

        self.model = model
        self.model.to("cuda")
        self.model.eval()

    def forward(self, batch_dict, iters=32):
        predictions = defaultdict(list)
        for stereo_pair in batch_dict["stereo_video"]:
            left_image_rgb = stereo_pair[None, 0].cuda()
            right_image_rgb = stereo_pair[None, 1].cuda()

            padder = self.igev_stereo_utils.InputPadder(left_image_rgb.shape, divis_by=32)
            left_image_rgb, right_image_rgb = padder.pad(
                left_image_rgb, right_image_rgb
            )

            with autocast(enabled=self.args.mixed_precision):
                disp = - self.model.forward(
                    left_image_rgb,
                    right_image_rgb,
                    iters=iters,
                    test_mode=True,
                )
            disp = padder.unpad(disp)
            predictions["disparity"].append(disp)
        predictions["disparity"] = (
            torch.stack(predictions["disparity"]).squeeze(1).abs()
        )
        return predictions

    def forward_stabilizer(self, batch_dict, model_stabilizer, iters=32):
        predictions = defaultdict(list)
        for stereo_pair in batch_dict["stereo_video"]:
            left_image_rgb = stereo_pair[None, 0].cuda()
            right_image_rgb = stereo_pair[None, 1].cuda()

            padder = self.igev_stereo_utils.InputPadder(left_image_rgb.shape, divis_by=32)
            left_image_rgb, right_image_rgb = padder.pad(
                left_image_rgb, right_image_rgb
            )

            with autocast(enabled=self.args.mixed_precision):
                disp = - self.model.forward(
                    left_image_rgb,
                    right_image_rgb,
                    iters=iters,
                    test_mode=True,
                )
            disp = padder.unpad(disp)
            predictions["disparity"].append(disp)
        predictions["disparity"] = (
            torch.stack(predictions["disparity"], dim=1)
        )

        disparities = model_stabilizer.forward_batch(batch_dict["stereo_video"][:, 0].cuda(),
                                                     predictions["disparity"].squeeze(0),
                                                     kernel_size=self.kernel_size)

        predictions["disparity"] = disparities.squeeze(1).abs()
        return predictions
