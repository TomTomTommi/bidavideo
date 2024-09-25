#!/bin/bash

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH

# RAFTStereo + BiDAStabilizer
python demo.py --model_name raftstereo --stabilizer \
 --ckpt ./checkpoints/raftstereo_robust/raftstereo_robust.pth \
 --stabilizer_ckpt ./checkpoints/raftstereo_stabilizer_robust/raftstereo_stabilizer_robust.pth \
 --path ./demo_video/ --output_path ./demo_output/RAFTStereo_BiDAStabilizer/ --save_png

 # IGEVStereo + BiDAStabilizer
python demo.py --model_name igevstereo --stabilizer \
 --ckpt ./checkpoints/igevstereo_robust/igevstereo_robust.pth \
 --stabilizer_ckpt ./checkpoints/igevstereo_stabilizer_robust/igevstereo_stabilizer_robust.pth \
 --path ./demo_video/ --output_path ./demo_output/IGEVStereo_BiDAStabilizer/ --save_png

# BiDAStereo
python demo.py --model_name bidastereo --ckpt ./checkpoints/bidastereo_robust/bidastereo_robust.pth \
         --path ./demo_video/ --output_path ./demo_output/BiDAStereo/ --save_png

# RAFTStereo
python demo.py --model_name raftstereo --ckpt ./checkpoints/raftstereo_robust/raftstereo_robust.pth \
         --path ./demo_video/ --output_path ./demo_output/RAFTStereo/ --save_png

# IGEVStereo
python demo.py --model_name igevstereo --ckpt ./checkpoints/igevstereo_robust/igevstereo_robust.pth \
         --path ./demo_video/ --output_path ./demo_output/IGEVStereo/ --save_png
