#!/bin/bash

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH

######################################## RAFTStereo SF ########################################

python ./evaluation/evaluate.py --config-name eval_sintel_clean \
MODEL.model_name=RAFTStereoModel \
MODEL.RAFTStereoModel.model_weights=./checkpoints/raftstereo_sf/raftstereo_sf.pth \
stabilizer_ckpt=./checkpoints/raftstereo_stabilizer_sf/raftstereo_stabilizer_sf.pth
exp_dir=./outputs/raftstereo_stabilizer_sf_sintel_clean

python ./evaluation/evaluate.py --config-name eval_sintel_final \
MODEL.model_name=RAFTStereoModel \
MODEL.RAFTStereoModel.model_weights=./checkpoints/raftstereo_sf/raftstereo_sf.pth \
stabilizer_ckpt=./checkpoints/raftstereo_stabilizer_sf/raftstereo_stabilizer_sf.pth
exp_dir=./outputs/raftstereo_stabilizer_sf_sintel_final

python ./evaluation/evaluate.py --config-name eval_dynamic_replica \
MODEL.model_name=RAFTStereoModel \
MODEL.RAFTStereoModel.model_weights=./checkpoints/raftstereo_sf/raftstereo_sf.pth \
stabilizer_ckpt=./checkpoints/raftstereo_stabilizer_sf/raftstereo_stabilizer_sf.pth
exp_dir=./outputs/raftstereo_stabilizer_sf_dynamic_replica

######################################## RAFTStereo SF + DR #####################################

python ./evaluation/evaluate.py --config-name eval_sintel_clean \
MODEL.model_name=RAFTStereoModel \
MODEL.RAFTStereoModel.model_weights=./checkpoints/raftstereo_sf_dr/raftstereo_sf_dr.pth \
stabilizer_ckpt=./checkpoints/raftstereo_stabilizer_sf_dr/raftstereo_stabilizer_sf_dr.pth
exp_dir=./outputs/raftstereo_stabilizer_sf_dr_sintel_clean

python ./evaluation/evaluate.py --config-name eval_sintel_final \
MODEL.model_name=RAFTStereoModel \
MODEL.RAFTStereoModel.model_weights=./checkpoints/raftstereo_sf_dr/raftstereo_sf_dr.pth \
stabilizer_ckpt=./checkpoints/raftstereo_stabilizer_sf_dr/raftstereo_stabilizer_sf_dr.pth
exp_dir=./outputs/raftstereo_stabilizer_sf_dr_sintel_final

python ./evaluation/evaluate.py --config-name eval_dynamic_replica \
MODEL.model_name=RAFTStereoModel \
MODEL.RAFTStereoModel.model_weights=./checkpoints/raftstereo_sf_dr/raftstereo_sf_dr.pth \
stabilizer_ckpt=./checkpoints/raftstereo_stabilizer_sf_dr/raftstereo_stabilizer_sf_dr.pth
exp_dir=./outputs/raftstereo_stabilizer_sf_dr_dynamic_replica

######################################## RAFTStereo SF + KITTI ##################################

python ./evaluation/evaluate.py --config-name eval_kittidepth \
MODEL.model_name=RAFTStereoModel \
MODEL.RAFTStereoModel.model_weights=./checkpoints/raftstereo_sf_kitti/raftstereo_sf_k.pth \
stabilizer_ckpt=./checkpoints/raftstereo_stabilizer_sf_kitti/raftstereo_stabilizer_sf_k.pth
exp_dir=./outputs/raftstereo_stabilizer_sf_k_kitti

######################################## RAFTStereo SF + Infinigen ##############################

python ./evaluation/evaluate.py --config-name eval_infinigensv \
MODEL.model_name=RAFTStereoModel \
MODEL.RAFTStereoModel.kernel_size=40 \
MODEL.RAFTStereoModel.model_weights=./checkpoints/raftstereo_sf_infinigen/raftstereo_sf_isv.pth \
stabilizer_ckpt=./checkpoints/raftstereo_sf_infinigen/raftstereo_stabilizer_sf_isv.pth
exp_dir=./outputs/raftstereo_stabilizersf_isv_infinigen

######################################## RAFTStereo Robust ######################################

python ./evaluation/evaluate.py --config-name eval_infinigensv \
MODEL.model_name=RAFTStereoModel \
MODEL.RAFTStereoModel.kernel_size=40 \
MODEL.RAFTStereoModel.model_weights=./checkpoints/raftstereo_robust/raftstereo_robust.pth \
stabilizer_ckpt=./checkpoints/raftstereo_stabilizer_robust/raftstereo_stabilizer_robust.pth \
exp_dir=./outputs/raftstereo_stabilizer_robust_InfinigenSV

python ./evaluation/evaluate.py --config-name eval_kittidepth \
MODEL.model_name=RAFTStereoModel \
MODEL.RAFTStereoModel.model_weights=./logging/raftstereo_robust/model_raftstereo_007507.pth \
stabilizer_ckpt=./logging/raftstereo_stabilizer_robust/model_raftstereo_stabilizer_015016.pth \
exp_dir=./outputs/raftstereo_stabilizer_robust_kitti

python ./evaluation/evaluate.py --config-name eval_dynamic_replica \
MODEL.model_name=RAFTStereoModel \
MODEL.RAFTStereoModel.kernel_size=40 \
MODEL.RAFTStereoModel.model_weights=./logging/raftstereo_robust/model_raftstereo_007507.pth \
stabilizer_ckpt=./logging/raftstereo_stabilizer_robust/model_raftstereo_stabilizer_015016.pth \
exp_dir=./outputs/raftstereo_stabilizer_robust_dynamic_replica

######################################## IGEVStereo SF ########################################

python ./evaluation/evaluate.py --config-name eval_sintel_clean \
MODEL.model_name=IGEVStereoModel \
MODEL.IGEVStereoModel.model_weights=./checkpoints/igevstereo_sf/igevstereo_sf.pth \
stabilizer_ckpt=./checkpoints/igevstereo_stabilizer_sf/igevstereo_stabilizer_sf.pth
exp_dir=./outputs/igevstereo_stabilizer_sf_sintel_clean

python ./evaluation/evaluate.py --config-name eval_sintel_final \
MODEL.model_name=IGEVStereoModel \
MODEL.IGEVStereoModel.model_weights=./checkpoints/igevstereo_sf/igevstereo_sf.pth \
stabilizer_ckpt=./checkpoints/igevstereo_stabilizer_sf/igevstereo_stabilizer_sf.pth
exp_dir=./outputs/igevstereo_stabilizer_sf_sintel_final

python ./evaluation/evaluate.py --config-name eval_dynamic_replica \
MODEL.model_name=RAFTStereoModel \
MODEL.IGEVStereoModel.model_weights=./checkpoints/igevstereo_sf/igevstereo_sf.pth \
stabilizer_ckpt=./checkpoints/igevstereo_stabilizer_sf/igevstereo_stabilizer_sf.pth
exp_dir=./outputs/igevstereo_stabilizer_sf_dynamic_replica

######################################## IGEVStereo SF + DR ###################################

python ./evaluation/evaluate.py --config-name eval_sintel_clean \
MODEL.model_name=IGEVStereoModel \
MODEL.IGEVStereoModel.model_weights=./checkpoints/igevstereo_sf_dr/igevstereo_sf_dr.pth \
stabilizer_ckpt=./checkpoints/igevstereo_stabilizer_sf_dr/igevstereo_stabilizer_sf_dr.pth
exp_dir=./outputs/igevstereo_stabilizer_sf_dr_sintel_clean

python ./evaluation/evaluate.py --config-name eval_sintel_final \
MODEL.model_name=IGEVStereoModel \
MODEL.IGEVStereoModel.model_weights=./checkpoints/igevstereo_sf_dr/igevstereo_sf_dr.pth \
stabilizer_ckpt=./checkpoints/igevstereo_stabilizer_sf_dr/igevstereo_stabilizer_sf_dr.pth
exp_dir=./outputs/igevstereo_stabilizer_sf_dr_sintel_final

python ./evaluation/evaluate.py --config-name eval_dynamic_replica \
MODEL.model_name=IGEVStereoModel \
MODEL.IGEVStereoModel.kernel_size=40 \
MODEL.IGEVStereoModel.model_weights=./checkpoints/igevstereo_sf_dr/igevstereo_sf_dr.pth \
stabilizer_ckpt=./checkpoints/igevstereo_stabilizer_sf_dr/igevstereo_stabilizer_sf_dr.pth
exp_dir=./outputs/igevstereo_stabilizer_sf_dr_dynamic_replica

######################################## IGEVStereo SF + KITTI ##################################

python ./evaluation/evaluate.py --config-name eval_kittidepth \
MODEL.model_name=IGEVStereoModel \
MODEL.IGEVStereoModel.model_weights=./checkpoints/igevstereo_sf_kitti/igevstereo_sf_k.pth \
stabilizer_ckpt=./checkpoints/igevstereo_stabilizer_sf_kitti/igevstereo_stabilizer_sf_k.pth \
exp_dir=./outputs/igevstereo_stabilizer_sf_k_kitti

######################################## IGEVStereo SF + Infinigen ##############################

python ./evaluation/evaluate.py --config-name eval_infinigensv \
MODEL.model_name=IGEVStereoModel \
MODEL.IGEVStereoModel.kernel_size=40 \
MODEL.IGEVStereoModel.model_weights=./checkpoints/igevstereo_sf_infinigen/igevstereo_sf_isv.pth \
stabilizer_ckpt=./checkpoints/igevstereo_stabilizer_sf_infinigen/igevstereo_stabilizer_sf_isv.pth \
exp_dir=./outputs/igevstereo_stabilizer_sf_isv_infinigen

######################################## IGEVStereo Robust ######################################

python ./evaluation/evaluate.py --config-name eval_infinigensv \
MODEL.model_name=IGEVStereoModel \
MODEL.IGEVStereoModel.kernel_size=40 \
MODEL.IGEVStereoModel.model_weights=./checkpoints/igevstereo_robust/igevstereo_robust.pth \
stabilizer_ckpt=./checkpoints/igevstereo_stabilizer_robust/igevstereo_stabilizer_robust.pth \
exp_dir=./outputs/igevstereo_stabilizer_robust_InfinigenSV

python ./evaluation/evaluate.py --config-name eval_kittidepth \
MODEL.model_name=IGEVStereoModel \
MODEL.IGEVStereoModel.model_weights=./checkpoints/igevstereo_robust/igevstereo_robust.pth \
stabilizer_ckpt=./checkpoints/igevstereo_stabilizer_robust/igevstereo_stabilizer_robust.pth \
exp_dir=./outputs/igevstereo_stabilizer_robust_kitti

python ./evaluation/evaluate.py --config-name eval_dynamic_replica \
MODEL.model_name=IGEVStereoModel \
MODEL.IGEVStereoModel.kernel_size=40 \
MODEL.IGEVStereoModel.model_weights=./checkpoints/igevstereo_robust/igevstereo_robust.pth \
stabilizer_ckpt=./checkpoints/igevstereo_stabilizer_robust/igevstereo_stabilizer_robust.pth \
exp_dir=./outputs/igevstereo_stabilizer_robust_dynamic_replica

######################################## SouthKensington ########################################

python ./evaluation/evaluate.py --config-name eval_southkensington \
MODEL.model_name=RAFTStereoModel \
MODEL.RAFTStereoModel.kernel_size=40 \
MODEL.RAFTStereoModel.model_weights=./checkpoints/raftstereo_robust/raftstereo_robust.pth \
stabilizer_ckpt=./checkpoints/raftstereo_stabilizer_robust/raftstereo_stabilizer_robust.pth \
exp_dir=./outputs/raftstereo_stabilizer_SouthKensingtonIndoor/

python ./evaluation/evaluate.py --config-name eval_southkensington \
MODEL.model_name=IGEVStereoModel \
MODEL.IGEVStereoModel.kernel_size=40 \
MODEL.IGEVStereoModel.model_weights=./checkpoints/igevstereo_robust/igevstereo_robust.pth \
stabilizer_ckpt=./checkpoints/igevstereo_stabilizer_robust/igevstereo_stabilizer_robust.pth \
exp_dir=./outputs/igevstereo_stabilizer_SouthKensingtonIndoor/
