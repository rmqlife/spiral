#!/bin/bash
conda create -n spiral python=3.7 numpy==1.16 jupyter matplotlib gast=0.2.2 astor
pip install opencv-python
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
