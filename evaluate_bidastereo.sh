#!/bin/bash

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH

# evaluate on [sintel, dynamicreplica， kittidepth， infinigensv， southkensington]

python ./evaluation/evaluate.py --config-name eval_sintel_clean \
MODEL.model_name=BiDAStereoModel \
MODEL.BiDAStereoModel.kernel_size=50 \
MODEL.BiDAStereoModel.type=bidastereo \
MODEL.BiDAStereoModel.model_weights=./checkpoints/bidastereo_sf_dr/bidastereo_sf_dr.pth

python ./evaluation/evaluate.py --config-name eval_sintel_final \
MODEL.model_name=BiDAStereoModel \
MODEL.BiDAStereoModel.kernel_size=50 \
MODEL.BiDAStereoModel.type=bidastereo \
MODEL.BiDAStereoModel.model_weights=./checkpoints/bidastereo_sf_dr/bidastereo_sf_dr.pth

python ./evaluation/evaluate.py --config-name eval_dynamic_replica \
MODEL.model_name=BiDAStereoModel \
MODEL.BiDAStereoModel.type=bidastereo \
MODEL.BiDAStereoModel.model_weights=./checkpoints/bidastereo_robust/bidastereo_robust.pth

python ./evaluation/evaluate.py --config-name eval_kittidepth \
MODEL.model_name=BiDAStereoModel \
MODEL.BiDAStereoModel.kernel_size=40 \
MODEL.BiDAStereoModel.type=bidastereo \
MODEL.BiDAStereoModel.model_weights=./checkpoints/bidastereo_robust/bidastereo_robust.pth

python ./evaluation/evaluate.py --config-name eval_infinigensv \
MODEL.model_name=BiDAStereoModel \
MODEL.BiDAStereoModel.kernel_size=20 \
MODEL.BiDAStereoModel.model_weights=./checkpoints/bidastereo_robust/bidastereo_robust.pth

python ./evaluation/evaluate.py --config-name eval_southkensington \
MODEL.model_name=BiDAStereoModel \
MODEL.BiDAStereoModel.type=bidastereo \
MODEL.BiDAStereoModel.model_weights=./checkpoints/bidastereo_robust/bidastereo_robust.pth \
exp_dir=./outputs/bidastereo_SouthKensingtonIndoor/dynamic/
