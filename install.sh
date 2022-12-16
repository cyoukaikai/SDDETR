#!/usr/bin/env bash



git clone https://github.com/cyoukaikai/SDDETR.git

cd SDDETR
chmod +x *.sh


mv pip.conf ~/.pip/pip.conf
mv watch_gpu.sh ../.
mv coco.sh ../.

./install_sd.sh
./install_yun_coco.sh

# MOVE the 
mv SDDETR/DN-DETR ../DN-DETR

cd DN-DETR

echo 8b14dd204de425f8a0c700ab58d00ec9ce60db4b
# tmux new -s train
wandb login
