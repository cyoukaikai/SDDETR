#!/usr/bin/env bash

# export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#echo "export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}" >> ~/.bashrc
#echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc

git clone https://github.com/cyoukaikai/SDDETR.git

cd SDDETR

mv pip.conf ~/.pip/pip.conf
mv condarc.sh ~/.condarc



chmod +x *.sh




conda create -n detr python=3.7 -y
echo "conda activate detr" >> ~/.bashrc
conda activate detr




mv watch_gpu.sh ../.
mv coco.sh ../.

./install_sd.sh
./install_yun_coco.sh

pip install opencv-python-headless==4.5.2.52
# MOVE the 
cd ../
mv SDDETR/DN-DETR ../DN-DETR

cd DN-DETR

echo 8b14dd204de425f8a0c700ab58d00ec9ce60db4b
# tmux new -s train
wandb login
