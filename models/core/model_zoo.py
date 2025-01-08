# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from bidavideo.models.bidastereo_model import BiDAStereoModel
from pytorch3d.implicitron.tools.config import get_default_args
try:
    from bidavideo.models.raft_stereo_model import RAFTStereoModel
    from bidavideo.models.igev_stereo_model import IGEVStereoModel
    MODELS = [RAFTStereoModel, IGEVStereoModel, BiDAStereoModel]
except:
    MODELS = [BiDAStereoModel]

_MODEL_NAME_TO_MODEL = {model_cls.__name__: model_cls for model_cls in MODELS}
_MODEL_CONFIG_NAME_TO_DEFAULT_CONFIG = {}
for model_cls in MODELS:
    _MODEL_CONFIG_NAME_TO_DEFAULT_CONFIG[
        model_cls.MODEL_CONFIG_NAME
    ] = get_default_args(model_cls)

MODEL_NAME_NONE = "NONE"


def model_zoo(model_name: str, **kwargs):
    if model_name.upper() == MODEL_NAME_NONE:
        return None

    model_cls = _MODEL_NAME_TO_MODEL.get(model_name)

    if model_cls is None:
        raise ValueError(f"No such model name: {model_name}")

    model_cls_params = {}
    if "model_zoo" in getattr(model_cls, "__dataclass_fields__", []):
        model_cls_params["model_zoo"] = model_zoo
    print(
        f"{model_cls.MODEL_CONFIG_NAME} model configs:",
        kwargs.get(model_cls.MODEL_CONFIG_NAME),
    )
    return model_cls(**model_cls_params, **kwargs.get(model_cls.MODEL_CONFIG_NAME, {}))


def get_all_model_default_configs():
    return copy.deepcopy(_MODEL_CONFIG_NAME_TO_DEFAULT_CONFIG)
