#!/bin/bash

cd /home/mscs/houminqiu2/SFSDNet
# 使用 $@ 转发 condor arguments 中的命令行参数
/home/mscs/houminqiu2/miniconda3/envs/VIC/bin/python3 train_p2r.py "$@"