#!/usr/bin/env bash

# chmod +x coco.sh

git clone https://github.com/cyoukaikai/SDDETR.git

mv SDDETR/* . 

rm -rf SDDETR

mkdir DN-DETR
mv DN-DETR.zip DN-DETR
cd DN-DETR
unzip DN-DETR.zip 


cd smrc
pip install -r requirements.txt

cd ../
pip install -r requirements.txt


pip install einops
pip install -U openmim
mim install mmcv-full
pip install mmdet

pip install wandb

touch ~/.tmux.conf
echo "setw -g mouse on" >> ~/.tmux.conf
tmux source-file ~/.tmux.conf


cd models/dn_dab_deformable_detr/ops
python setup.py build install
# python test.py
cd ../../..


# sftp://20.172.254.9/
#  ssh -i ~/sddetr_key.pem kaikai@20.172.254.9
wandb login
