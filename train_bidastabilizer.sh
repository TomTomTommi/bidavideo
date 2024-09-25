#!/bin/bash

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH

# RAFTStereo
python train_bidastabilizer.py --name raftstereo_stabilizer --batch_size 1 \
 --spatial_scale -0.2 0.4 --image_size 192 384 --saturation_range 0 1.4 --num_steps 20000  \
 --restore_ckpt checkpoints/raftstereo_sf/raftstereo_sf.pth --ckpt_path logging/raftstereo_stabilizer_sf/ \
 --sample_len 8 --lr 0.0001 --train_iters 22 --valid_iters 32    \
 --num_workers 1 --save_freq 100 --train_datasets things monkaa driving

# IGEVStereo
python train_bidastabilizer.py --name igevstereo_stabilizer --batch_size 1 \
 --spatial_scale -0.2 0.4 --image_size 192 384 --saturation_range 0 1.4 --num_steps 20000  \
 --restore_ckpt checkpoints/igevstereo_sf/igevstereo_sf.pth --ckpt_path logging/igevstereo_stabilizer_sf/ \
 --sample_len 8 --lr 0.0001 --train_iters 22 --valid_iters 32    \
 --num_workers 1 --save_freq 100 --train_datasets things monkaa driving