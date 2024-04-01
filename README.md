# Robust Cooperative Multi-agent Reinforcement Learning via Multi-view Message Certification

This repository contains the implementation of CroMAC, based on PyTorch. 

## 1. Getting started

Use the install script to install the python environment:

```shell
conda create -n CroMAC python=3.7 -y
conda activate CroMAC
pip install torch torchvision torchaudio
pip install sacred numpy scipy matplotlib seaborn pyyaml pygame pytest probscale imageio snakeviz tensorboard-logger tensorboard tensorboardx
cd CroMAC
pip install -e qplex_smac/
pip install -e lb-foraging-master/
pip install gym==0.21.0
pip install importlib-metadata==4.13.0
bash install_sc2.sh
unzip map.zip -d $SC2PATH/Maps/SMAC_MAPS
```

## 2. Run an experiment
All the experiments can be run with the unified entrance file `src/main.py` with customized arguments.

Training scripts are also provided in the `runalgo.sh` script.

