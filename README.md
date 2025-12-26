# venv create
```
conda create -n vsnet python==3.9
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda activate vsnet
pip install torch_scatter-2.1.0+pt112cu116-cp39-cp39-linux_x86_64.whl
pip install spconv-cu116
pip install -r requirements.txt

# install detectron2
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

#install open_clip_torch
pip install open_clip_torch

# install cuda11.6
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
sudo sh cuda_11.6.0_510.39.01_linux.run
sudo vim ~/.bashrc

export PATH=/usr/local/cuda-11.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-11.6

source ~/.bashrc
sudo ldconfig
```


# SemanticKitti DataSets
## Dataset Structure

```
SemanticKitti/
├── dataset
│   └── sequences					# 21 sequences
│   		├── 00					# 00~07 + 09~10: training split; 08: val split; Other：test split. For test split, there is no labels folder.
│   		│	├── image_2			# Left Camrea
│   		│	├── image_3			# Right Camera
│   		│	├── instance		# Download as shown in the following. Format: XXX.bin
│   		│	├── labels			# Format: XXX.label
│   		│	├── velodyne		# Format: XXX.bin
│   		│	├── calib.txt
│   		│	├── poses.txt
│   		│	└── times.txt
│   		├── 01
│   		├── 02
│   		├── ...
│   		└── 21
└── instance_path.pkl				# Download pkl file as shown in the following.
```

## train
```
conda activate vsnet
export OMP_NUM_THREADS=6
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
nohup python -m torch.distributed.run --nproc_per_node=6 train.py -c configs/pa_po_kitti_trainval.yaml -l runs/trainkitti7/train.log  > runs/trainkitti7/train_nohup.log 2>&1 & 
```