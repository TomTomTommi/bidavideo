# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import ClassVar
import torch
from pytorch3d.implicitron.tools.config import Configurable

class BiDAStereoModel(Configurable, torch.nn.Module):

    MODEL_CONFIG_NAME: ClassVar[str] = "BiDAStereoModel"
    model_weights: str = "./checkpoints/bidastereo_sf/bidastereo_sf.pth"
    type: str = "bidastereo"
    kernel_size: int = 20

    def __post_init__(self):
        super().__init__()

        self.mixed_precision = False

        if self.type == 'bidastereo':
            from bidavideo.models.core.bidastereo import BiDAStereo
            self.iter = 20
            model = BiDAStereo(
                mixed_precision=self.mixed_precision,
            )
        else:
            raise ValueError("Wrong Model!")

        state_dict = torch.load(self.model_weights, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            state_dict = {"module." + k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)

        self.model = model
        self.model.to("cuda")
        self.model.eval()

    def forward(self, batch_dict, iters=20):
        return self.model.forward_batch_test(
            batch_dict, kernel_size=self.kernel_size, iters=self.iter
        )
